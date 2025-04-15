from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Response from an agent.

    Attributes:
        type: Type of response (text, tool_call, error)
        sender_name: Name of sending agent
        receiver_name: Name of receiving agent
        content: Response content
        raw_response: Raw response object (optional)
    """

    type: Literal["text", "tool_call", "error"] = Field(
        description="Type of response (text, tool_call, error)"
    )
    sender_name: str = Field(description="Name of sending agent")
    receiver_name: str = Field(description="Name of receiving agent")
    content: str = Field(description="Response content")
    raw_response: Any | None = Field(default=None, description="Raw response object")

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
