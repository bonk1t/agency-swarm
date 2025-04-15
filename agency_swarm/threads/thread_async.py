from __future__ import annotations

import logging
import threading

from openai.types.responses.response_create_params import ToolChoice

from agency_swarm.messages.agent_response import AgentResponse
from agency_swarm.threads.thread import Thread

logger = logging.getLogger(__name__)


class ThreadAsync(Thread):
    """Asynchronous version of Thread that supports non-blocking tool execution.

    This class extends Thread to provide non-blocking agent communication.
    The key differences from Thread are:
    1. Uses threading for non-blocking execution
    2. Provides status checking and result retrieval
    3. Maintains state for async tool execution
    """

    def __init__(self, *args, **kwargs):
        """Initialize ThreadAsync with proper attribute initialization.

        Ensures that all async-specific attributes are properly initialized
        before any tool calls or error handling can occur.
        """
        # Import Thread here to avoid circular import
        from agency_swarm.threads.thread import Thread

        if not isinstance(self, Thread):
            Thread.__init__(self, *args, **kwargs)

        # Initialize tracking attributes for async tool execution
        self._tool_calls_in_progress = []  # Track currently executing tools
        self._tool_results = {}  # Store results as they complete

        # Set async mode to tools_threading by default for this class
        self.async_mode = kwargs.get("async_mode", "tools_threading")

        logger.debug(f"Initialized ThreadAsync with async_mode={self.async_mode}")

        self.pythread = None
        self.response = None
        self._is_processing = False
        self._error = None

    def worker(
        self,
        message: str,
        message_files: list[str] = None,
        recipient_agent=None,
        additional_instructions: str = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ):
        """Worker method for async processing."""
        import asyncio

        # Create event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Set processing state
        self._is_processing = True
        self._tool_calls_in_progress = []

        try:
            # Get response using base Thread implementation
            response: AgentResponse = loop.run_until_complete(
                self.get_response(
                    message=message,
                    message_files=message_files,
                    recipient_agent=recipient_agent,
                    additional_instructions=additional_instructions,
                    tool_choice=tool_choice,
                    parent_run_id=parent_run_id,
                )
            )

            # Store successful response and clear error
            self.response: AgentResponse = response
            self._error = None
            self._tool_calls_in_progress = []

        except Exception as e:
            # Store error response but keep previous successful response
            self._error = e
            self._tool_calls_in_progress = []
            if not self.response:  # Only create error response if no previous response
                self.response = AgentResponse.from_error(
                    e,
                    sender_name=self.sender.name,
                    receiver_name=self.recipient.name,
                )
        finally:
            # Only clear processing flag, keep response and error state
            self._is_processing = False
            loop.close()

    def get_response_async(
        self,
        message: str,
        message_files: list[str] = None,
        recipient_agent=None,
        additional_instructions: str = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ) -> AgentResponse:  # TODO: update this method to return AgentResponse
        """Start async processing of a message.

        Args:
            message: The message to send (string)
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Either "auto", "none", or a dict specifying a tool
            parent_run_id: Optional parent run ID for tracking

        Returns:
            dict: Message in Responses API format with type and content fields
        """
        if self._is_processing:
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Agent is busy processing a request. Please check status later.",
                    }
                ],
            }

        self.pythread = threading.Thread(
            target=self.worker,
            args=(
                message,
                message_files,
                recipient_agent,
                additional_instructions,
                tool_choice,
                parent_run_id,
            ),
        )

        self.pythread.start()
        return {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Task has started. You can check the status later.",
                }
            ],
        }

    def check_status(
        self,
    ) -> AgentResponse:  # TODO: update this method to return AgentResponse
        """Check status of async response.

        Returns:
            dict: Message in Responses API format indicating:
            - If agent is ready for new messages
            - If task is still processing (including current tool calls)
            - The response content if complete
            - Error message if task failed
        """
        if self._is_processing:
            if self._tool_calls_in_progress:
                tool_names = [t.function.name for t in self._tool_calls_in_progress]
                return {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Task is in progress. Currently executing tools: {', '.join(tool_names)}",
                        }
                    ],
                }
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Task is still in progress. Please try again later.",
                    }
                ],
            }

        if not self.response:
            return {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Agent is ready to receive a message."}
                ],
            }

        if self._error:
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task failed with error: {str(self._error)}",
                    }
                ],
            }

        if isinstance(self.response, AgentResponse):
            return {
                "role": "assistant",
                "content": [{"type": "text", "text": self.response.content}],
            }

        return {
            "role": "system",
            "content": [
                {"type": "text", "text": "Unknown response state. Please try again."}
            ],
        }

    def is_busy(self) -> bool:
        """Check if the agent is currently processing a request."""
        return self._is_processing

    def get_error(self) -> Exception | None:
        """Get the last error that occurred, if any."""
        return self._error

    def clear_state(self):
        """Clear the thread's state for a new request."""
        self.response = None
        self._error = None
        self._is_processing = False
        self._tool_calls_in_progress = []
        self._tool_results = {}
