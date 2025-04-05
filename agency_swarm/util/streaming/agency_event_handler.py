from abc import ABC

from openai.lib.streaming import AssistantEventHandler


class AgencyEventHandler(AssistantEventHandler, ABC):
    agent_name = None
    recipient_agent_name = None
    agent = None
    recipient_agent = None

    @classmethod
    def on_text_delta(cls, delta: str, sender_name: str):
        """Called when a text delta is received."""
        pass

    @classmethod
    def on_tool_call_created(cls, tool_call: dict):
        """Called when a tool call is created.

        Args:
            tool_call: Tool call object from Responses API with structure:
            {
                "type": "function_call",
                "id": str,
                "call_id": str,
                "name": str,
                "arguments": str
            }
        """
        pass

    @classmethod
    def on_tool_call_completed(cls, tool_call: dict, result: str):
        """Called when a tool call is completed.

        Args:
            tool_call: The original tool call object
            result: The result returned by the tool
        """
        pass

    @classmethod
    def on_tool_call_error(cls, tool_call: dict, error: str):
        """Called when a tool call results in an error.

        Args:
            tool_call: The original tool call object
            error: Error message
        """
        pass

    @classmethod
    def on_response_completed(cls):
        """Called when the full response is completed."""
        pass

    @classmethod
    def on_all_streams_end(cls):
        """Called when all streams have ended."""
        pass

    @classmethod
    def set_agent(cls, value):
        cls.agent = value
        cls.agent_name = value.name if value else None

    @classmethod
    def set_recipient_agent(cls, value):
        cls.recipient_agent = value
        cls.recipient_agent_name = value.name if value else None
