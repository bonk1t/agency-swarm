from openai import BadRequestError
from pydantic import Field

from .send_message_base import SendMessageBase


class SendMessageSwarm(SendMessageBase):
    """Use this tool to facilitate direct, synchronous communication between specialized agents within your agency. When you send a message using this tool, you receive a response exclusively from the designated recipient agent. To continue the dialogue, invoke this tool again with the desired recipient agent and your follow-up message. Remember, communication here is synchronous; the recipient agent won't perform any tasks post-response. You are responsible for relaying the recipient agent's responses back to the user, as the user does not have direct access to these replies. Keep engaging with the tool for continuous interaction until the task is fully resolved. Do not send more than 1 message to the same recipient agent at the same time."""

    message: str = Field(
        ...,
        description="Specify the task required for the recipient agent to complete. Focus on clarifying what the task entails, rather than providing exact instructions. Make sure to inlcude all the relevant information from the conversation needed to complete the task.",
    )

    class ToolConfig:
        # set output as result because the communication will be finished after this tool is called
        output_as_result: bool = True
        one_call_at_a_time: bool = True

    def run(self):
        # get main thread
        thread = self._get_main_thread()

        # get recipient agent from thread
        recipient_agent = self._get_recipient_agent()

        # submit tool output
        try:
            thread.submit_tool_outputs(
                tool_outputs=[
                    {
                        "tool_call_id": self._tool_call.id,
                        "output": "The request has been routed. You are now a "
                        + recipient_agent.name
                        + " agent. Please assist the user further with their request.",
                    }
                ],
                poll=False,
            )
        except BadRequestError as e:
            raise Exception(
                "You can only call this tool by itself. Do not use any other tools together with this tool."
            )

        try:
            # cancel run
            thread.cancel_run()

            # change recipient agent in thread
            thread.recipient_agent = recipient_agent

            # change recipient agent in gradio dropdown
            if self._event_handler:
                if hasattr(self._event_handler, "change_recipient_agent"):
                    self._event_handler.change_recipient_agent(self.recipient.value)

            # continue conversation with the new recipient agent
            if self._event_handler:
                message = thread.get_response_stream(
                    message=None,
                    recipient_agent=recipient_agent,
                    event_handler=self._event_handler,
                    parent_run_id=self._tool_call.id,
                )
            else:
                message = thread.get_response(
                    message=None,
                    recipient_agent=recipient_agent,
                    parent_run_id=self._tool_call.id,
                )

            return message or ""
        except Exception as e:
            # we need to catch errors beucase tool outputs are already submitted
            print("Error in SendMessageSwarm: ", e)
            return str(e)
