from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List, Union

from openai.types.responses.response_create_params import ToolChoice

from agency_swarm.messages.agent_response import AgentResponse
from agency_swarm.threads import Thread
from agency_swarm.user import User

if TYPE_CHECKING:
    from agency_swarm.agents import Agent


class ThreadAsync(Thread):
    def __init__(self, agent: Union[Agent, User], recipient_agent: "Agent"):
        super().__init__(agent, recipient_agent)
        self.pythread = None
        self.response = None
        self._is_processing = False
        self._error = None

    def worker(
        self,
        message: str | list[dict],
        message_files: List[str] = None,
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

        try:
            self._is_processing = True
            self._error = None

            # Get response using base Thread implementation
            response = loop.run_until_complete(
                self.get_response(
                    message=message,
                    message_files=message_files,
                    recipient_agent=recipient_agent,
                    additional_instructions=additional_instructions,
                    tool_choice=tool_choice,
                    parent_run_id=parent_run_id,
                )
            )

            # Store the AgentResponse object
            self.response = response

        except Exception as e:
            self._error = e
            self.response = AgentResponse.from_error(
                e, sender_name=self.recipient_agent.name, receiver_name=self.agent.name
            )
        finally:
            self._is_processing = False
            loop.close()

    def get_response_async(
        self,
        message: str | list[dict],
        message_files: List[str] = None,
        recipient_agent=None,
        additional_instructions: str = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ) -> str:
        """Start async processing of a message.

        Args:
            message: The message to send (string or list of message dicts)
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Either "auto", "none", or a dict specifying a tool
            parent_run_id: Optional parent run ID for tracking

        Returns:
            Initial notification message
        """
        if self._is_processing:
            return "System Notification: 'Agent is busy processing a request. Please check status later.'"

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
        return (
            "System Notification: 'Task has started. You can check the status later.'"
        )

    def check_status(self) -> str:
        """Check status of async response.

        Returns:
            Status message indicating:
            - If agent is ready for new messages
            - If task is still processing
            - The response content if complete
            - Error message if task failed
        """
        if self._is_processing:
            return "System Notification: 'Task is still in progress. Please try again later.'"

        if not self.response:
            return "System Notification: 'Agent is ready to receive a message.'"

        if self._error:
            return f"System Notification: 'Task failed with error: {str(self._error)}'"

        # Return the actual response content
        if isinstance(self.response, AgentResponse):
            return f"{self.recipient_agent.name}'s Response: '{self.response.content}'"

        return "System Notification: 'Unknown response state. Please try again.'"

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
