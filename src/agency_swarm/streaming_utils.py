"""
Utilities for managing streaming events across nested agent calls.

This module provides infrastructure for collecting and forwarding events
from sub-agents to maintain full visibility during streaming operations.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def add_agent_name_to_event(event: Any, agent_name: str, caller_agent: str | None = None) -> Any:
    """Add agent name and caller to a streaming event.

    Args:
        event: The streaming event (dict or object)
        agent_name: Name of the agent to add to the event
        caller_agent: Name of the calling agent (None for user)

    Returns:
        The event with agent_name and caller_agent added
    """
    if isinstance(event, dict):
        event["agent_name"] = agent_name
        event["caller_agent"] = caller_agent
    elif hasattr(event, "__dict__"):
        # For object-like events, add as attributes
        event.agent_name = agent_name
        event.caller_agent = caller_agent

        # Try to enhance the repr/str by modifying the class itself
        try:
            original_class = event.__class__

            # Check if we've already enhanced this class
            if not hasattr(original_class, "_enhanced_repr_original"):
                # Store the original __repr__ method
                original_class._enhanced_repr_original = original_class.__repr__

                # Create a custom __repr__ method that shows all attributes
                def enhanced_repr(self):
                    # Get all attributes, including our new ones
                    attrs = []
                    for attr_name in sorted(dir(self)):
                        if not attr_name.startswith("_") and not callable(getattr(self, attr_name, None)):
                            try:
                                value = getattr(self, attr_name)
                                attrs.append(f"{attr_name}={repr(value)}")
                            except (AttributeError, TypeError):
                                pass

                    attrs_str = ", ".join(attrs)
                    return f"{original_class.__name__}({attrs_str})"

                # Replace the class __repr__ method
                original_class.__repr__ = enhanced_repr
                original_class.__str__ = enhanced_repr

        except Exception as e:
            logger.error(f"Could not add agent name to event: {e}")

    return event


@dataclass
class StreamingContext:
    """Context for managing event streaming across nested agent calls."""

    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_streaming: bool = True
    _merge_task: asyncio.Task | None = None

    async def put_event(self, event: Any) -> None:
        """Add an event to the queue."""
        await self.event_queue.put(event)

    async def get_event(self) -> Any:
        """Get an event from the queue."""
        return await self.event_queue.get()

    def stop(self) -> None:
        """Signal that streaming is complete."""
        self.event_queue.put_nowait(None)  # Sentinel value


class EventStreamMerger:
    """Merges events from multiple sources during streaming operations."""

    def __init__(self):
        self.streaming_context: StreamingContext | None = None

    @asynccontextmanager
    async def create_streaming_context(self):
        """Create a new streaming context for collecting events."""
        self.streaming_context = StreamingContext()
        try:
            yield self.streaming_context
        finally:
            if self.streaming_context:
                self.streaming_context.stop()
                self.streaming_context = None

    async def merge_streams(
        self,
        primary_stream: AsyncGenerator[Any],
        context: StreamingContext,
    ) -> AsyncGenerator[Any]:
        """
        Merge events from the primary stream and the context's event queue.

        This allows sub-agent events to be interleaved with the main agent's events.
        """
        # Create tasks for both sources
        primary_task = asyncio.create_task(self._consume_primary(primary_stream))
        queue_task = asyncio.create_task(self._consume_queue(context))

        pending = {primary_task, queue_task}

        try:
            while pending:
                # Wait for the first task to produce an event
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    try:
                        event = task.result()
                        if event is None:
                            # Stream ended
                            if task == primary_task:
                                logger.debug("Primary stream ended")
                                # Cancel queue task if primary is done
                                queue_task.cancel()
                                return
                            else:
                                logger.debug("Queue stream ended")
                                # Queue ended, but primary might still have events
                                continue
                        else:
                            yield event

                            # Restart the task to get the next event
                            if task == primary_task:
                                primary_task = asyncio.create_task(self._consume_primary(primary_stream))
                                pending.add(primary_task)
                            else:
                                queue_task = asyncio.create_task(self._consume_queue(context))
                                pending.add(queue_task)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.error(f"Error in stream merger: {e}")
                        raise
        finally:
            # Clean up any remaining tasks
            for task in pending:
                task.cancel()

    async def _consume_primary(self, stream: AsyncGenerator[Any]) -> Any:
        """Consume one event from the primary stream."""
        try:
            return await stream.__anext__()
        except StopAsyncIteration:
            return None

    async def _consume_queue(self, context: StreamingContext) -> Any:
        """Consume one event from the queue."""
        event = await context.get_event()
        return event  # None is sentinel for end


# Global instance for the agency
event_stream_merger = EventStreamMerger()
