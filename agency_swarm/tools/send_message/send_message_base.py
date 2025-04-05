from abc import ABC
from typing import TYPE_CHECKING, ClassVar, Union

from pydantic import Field, field_validator

from agency_swarm.threads.thread import Thread
from agency_swarm.tools import BaseTool

if TYPE_CHECKING:
    from agency_swarm.agents.agent import Agent


class SendMessageBase(BaseTool, ABC):
    recipient: str = Field(
        ...,
        description="Recipient agent that you want to send the message to. This field will be overriden inside the agency class.",
    )

    _agents_and_threads: ClassVar = None

    @field_validator("additional_instructions", mode="before", check_fields=False)
    @classmethod
    def validate_additional_instructions(cls, value):
        # previously the parameter was a list, now it's a string
        # add compatibility for old code
        if isinstance(value, list):
            return "\n".join(value)
        return value

    def _get_thread(self) -> Thread:
        return self._agents_and_threads[self._caller_agent.name][self.recipient.value]

    def _get_main_thread(self) -> Thread:
        return self._agents_and_threads["main_thread"]

    def _get_recipient_agent(self) -> "Agent":
        return self._agents_and_threads[self._caller_agent.name][
            self.recipient.value
        ].recipient_agent

    def _get_completion(self, message: Union[str, None] = None, **kwargs):
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
