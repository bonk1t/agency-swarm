from abc import ABC
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from agency_swarm.messages.agent_response import AgentResponse
from agency_swarm.threads.thread import Thread
from agency_swarm.tools import BaseTool
from agency_swarm.user import User

if TYPE_CHECKING:
    from agency_swarm.agents.agent import Agent


class SendMessageBase(BaseTool, ABC):
    recipient: str = Field(
        ...,
        description="Recipient agent that you want to send the message to. This field will be overriden inside the agency class.",
    )

    _threads: ClassVar[dict[str, Thread]] = (
        None  # it's just a pointer to the agency's threads (agency.threads)
    )
    # TODO: stop storing data in class attributes, it's not scalable

    def _get_thread(self) -> Thread:
        """Get the thread for communication with the recipient agent."""
        return self._threads[f"{self._caller_agent.name}->{self.recipient}"]

    def _get_user_thread(self) -> Thread:
        """Get the thread for communication with the user."""
        return self._threads[f"user->{self._caller_agent.name}"]

    def _get_recipient_agent(self) -> "Agent":
        """Get the recipient agent instance."""
        return self._threads[f"{self._caller_agent.name}->{self.recipient}"].recipient

    def _get_response(self, message: str | None = None, **kwargs) -> AgentResponse:
        thread = self._get_thread()

        if self.ToolConfig.async_mode == "threading":
            return thread.get_response_async(
                message=message,
                parent_run_id=self._tool_call.id,
                **kwargs,
            )
        else:
            if self._event_handler:
                return thread.get_response_stream(
                    message=message,
                    event_handler=self._event_handler,
                    parent_run_id=self._tool_call.id,
                    **kwargs,
                )
            else:
                return thread.get_response(
                    message=message,
                    parent_run_id=self._tool_call.id,
                    **kwargs,
                )
