from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from openai.types.responses import Response


@dataclass
class AgentResponse:
    """Response from an agent.

    Attributes:
        type: Type of response (text, tool_call, error)
        sender_name: Name of sending agent
        receiver_name: Name of receiving agent
        content: Response content
        raw_response: Raw response object
    """

    type: Literal["text", "tool_call", "error"]
    sender_name: str
    receiver_name: str
    content: str
    raw_response: Any = None

    @classmethod
    def from_response(
        cls, response: Response, sender_name: str, receiver_name: str
    ) -> "AgentResponse":
        """Create AgentResponse from OpenAI Response object."""
        # Check for tool calls
        for output_item in response.output:
            if output_item.type == "function_call":
                return cls.from_tool_call(
                    output_item, sender_name=sender_name, receiver_name=receiver_name
                )

        # Otherwise treat as text response
        return cls(
            type="text",
            sender_name=sender_name,
            receiver_name=receiver_name,
            content=response.output_text,
            raw_response=response,
        )

    @classmethod
    def from_tool_call(
        cls, tool_call: Any, sender_name: str, receiver_name: str
    ) -> "AgentResponse":
        """Create AgentResponse from tool call."""
        return cls(
            type="tool_call",
            sender_name=sender_name,
            receiver_name=receiver_name,
            content=f"Calling function {tool_call.function.name} with arguments: {tool_call.function.arguments}",
            raw_response=tool_call,
        )

    @classmethod
    def from_error(
        cls, error: Exception, sender_name: str, receiver_name: str
    ) -> "AgentResponse":
        """Create AgentResponse from error."""
        return cls(
            type="error",
            sender_name=sender_name,
            receiver_name=receiver_name,
            content=str(error),
            raw_response=error,
        )
