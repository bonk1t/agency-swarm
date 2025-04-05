from __future__ import annotations

import inspect
import json
import logging
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Type, Union

from openai import AsyncOpenAI
from openai.types.responses import Response
from openai.types.responses.response_create_params import ToolChoice

from agency_swarm.constants import MAX_AGENT_CALLS
from agency_swarm.messages import AgentResponse
from agency_swarm.user import User
from agency_swarm.util.files import get_file_purpose
from agency_swarm.util.oai import get_openai_client
from agency_swarm.util.streaming.agency_event_handler import AgencyEventHandler
from agency_swarm.util.tracking.tracking_manager import TrackingManager

if TYPE_CHECKING:
    from agency_swarm.agents import Agent

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in an agent's functions."""

    pass


class Thread:
    """
    A class representing a conversation thread between agents.
    """

    def __init__(
        self,
        agent: Union[Agent, User],
        recipient_agent: Agent,
        previous_response_id: str | None = None,
        messages: list[dict] | None = None,
        on_response: Callable[[AgentResponse], None] | None = None,
        on_tool_call: Callable[[AgentResponse], None] | None = None,
        on_error: Callable[[AgentResponse], None] | None = None,
    ):
        """Initialize a new thread.

        Args:
            agent: The sending agent or user
            recipient_agent: The receiving agent
            previous_response_id: Optional ID of previous response for chaining
            messages: Optional list of previous messages in the conversation
            on_response: Callback for text responses
            on_tool_call: Callback for tool calls
            on_error: Callback for errors
        """
        self.id = str(uuid.uuid4())
        self.messages = messages or []
        self._client = None
        self.shared_state = None
        self.tracking_manager = TrackingManager()

        self.agent = agent
        self.recipient_agent = recipient_agent
        self.previous_response_id = previous_response_id
        self.on_response = on_response
        self.on_tool_call = on_tool_call
        self.on_error = on_error

        # Track number of calls between agent pairs
        self._agent_call_counts = defaultdict(int)

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if not self._client:
            self._client = get_openai_client()
        return self._client

    def _get_thread_key(self) -> str:
        """Get unique key for current agent pair."""
        return f"{self.agent.name}->{self.recipient_agent.name}"

    def _prepare_tools(self) -> list[dict]:
        """Convert agent tools to OpenAI format."""
        if not self.recipient_agent.tools:
            return []

        tools = []
        for tool in self.recipient_agent.tools:
            tool_dict = {
                "type": "function",
                "name": tool.__name__,
                "description": tool.__doc__ or "",
                "parameters": tool.model_json_schema(),
            }
            tools.append(tool_dict)
        return tools

    async def _prepare_input(
        self, message: str | list[dict] | None, message_files: list[str] | None = None
    ) -> list[dict]:
        """Prepare input messages including conversation history.

        Formats new messages and combines them with existing conversation history.

        Args:
            message: New message to add to history
            message_files: Optional list of file IDs to attach

        Returns:
            List of messages including history

        Raises:
            ValueError: If message format is invalid
        """
        # Initialize messages list if None
        if self.messages is None:
            self.messages = []

        # Return existing messages if no new message
        if not message and not message_files:
            return self.messages.copy()

        # Convert string message to list format
        if isinstance(message, str):
            content = [{"type": "input_text", "text": message}]
            if message_files:
                for file_id in message_files:
                    content.append({"type": "input_file", "file_id": file_id})
            new_messages = [{"role": "user", "content": content}]
        elif isinstance(message, list):
            # Validate message format
            for msg in message:
                if (
                    not isinstance(msg, dict)
                    or "role" not in msg
                    or "content" not in msg
                ):
                    raise ValueError(
                        "Each message must be a dict with 'role' and 'content' keys"
                    )

                # Convert content to proper format if needed
                if isinstance(msg["content"], str):
                    msg["content"] = [{"type": "input_text", "text": msg["content"]}]
                elif isinstance(msg["content"], list):
                    # Already in proper format
                    pass
                else:
                    raise ValueError(
                        "Message content must be a string or list of content items"
                    )
            new_messages = message
        else:
            raise ValueError("Message must be a string or list of dicts")

        # Add new messages to history
        self.messages.extend(new_messages)

        # Return copy to avoid modifying history
        return self.messages.copy()

    async def get_response(
        self,
        message: str | list[dict] | None,
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ) -> AgentResponse:
        """Get a response from the recipient agent.

        Args:
            message: The message to send
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Either "auto", "none", "required" or a dict specifying a tool
            parent_run_id: Optional parent run ID for tracking

        Returns:
            AgentResponse object containing the response
        """
        if not recipient_agent:
            recipient_agent = self.recipient_agent

        # Set correct sender/receiver names based on message flow
        sender_name = "user" if isinstance(self.agent, User) else self.agent.name

        # The receiver should be 'user' if:
        # 1. The message originated from a user, OR
        # 2. The message is being sent directly to an agent from the user's perspective
        receiver_name = (
            "user"
            if (
                isinstance(self.agent, User)
                or (recipient_agent and isinstance(self.agent, User))
            )
            else self.agent.name
        )

        # Debug info with clear message flow
        print(
            f"RESPONSE:[ {sender_name} -> {recipient_agent.name} ] (receiver: {receiver_name})"
        )

        # Prepare input with history
        input_messages = await self._prepare_input(message, message_files)

        # Track start of interaction
        run_id = self.tracking_manager.start_run(
            message=message,
            sender_agent=sender_name,
            recipient_agent=recipient_agent.name,
            run_id=None,
            parent_run_id=parent_run_id,
            model=recipient_agent.model,
            temperature=recipient_agent.temperature,
        )

        try:
            # Create response with explicit instructions about maintaining context
            response = await self.client.responses.create(
                model=recipient_agent.model,
                input=input_messages,
                instructions=(additional_instructions or recipient_agent.instructions)
                + "\nMaintain and reference the conversation history in your responses.",
                tools=self._prepare_tools(),
                temperature=recipient_agent.temperature,
                previous_response_id=self.previous_response_id,
                tool_choice=tool_choice,
                stream=False,
            )

            # Update previous_response_id
            self.previous_response_id = response.id

            # Check for tool calls in output
            tool_calls = []
            for output_item in response.output:
                if output_item.type == "function_call":
                    tool_calls.append(output_item)
                elif output_item.type == "message":
                    # Add assistant's response to message history
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": output_item.content[0].text
                            if output_item.content
                            else "",
                        }
                    )

            # Handle tool calls if present
            if tool_calls:
                tool_outputs = await self._handle_tool_calls(
                    tool_calls=tool_calls,
                    response=response,
                    recipient_agent=recipient_agent,
                    event_handler=None,
                    parent_run_id=parent_run_id,
                )

                # Create agent response with tool outputs
                agent_response = AgentResponse(
                    type="tool_call",
                    sender_name=recipient_agent.name,
                    receiver_name=receiver_name,
                    content=tool_outputs,
                    raw_response=response,
                )
            else:
                # Create text response
                content = response.output[0].content[0].text if response.output else ""
                agent_response = AgentResponse(
                    type="text",
                    sender_name=recipient_agent.name,
                    receiver_name=receiver_name,
                    content=content,
                    raw_response=response,
                )

            # Track successful completion
            self.tracking_manager.end_run(agent_response, run_id)

            return agent_response

        except Exception as e:
            # Track error and re-raise
            self.tracking_manager.track_chain_error(e, run_id, parent_run_id)
            raise

    async def get_response_stream(
        self,
        message: str | list[dict] | None,
        event_handler: Type[AgencyEventHandler],
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ) -> AsyncIterator[AgentResponse]:
        """Streaming version of get_response that yields AgentResponse events."""
        if not recipient_agent:
            recipient_agent = self.recipient_agent

        # Debug info
        sender_name = "user" if isinstance(self.agent, User) else self.agent.name
        print(f"RESPONSE_STREAM:[ {sender_name} -> {recipient_agent.name} ]")

        # Prepare input
        input_messages = await self._prepare_input(message, message_files)

        # Track start of interaction
        run_id = self.tracking_manager.start_run(
            message=message,
            sender_agent=self.agent.name,
            recipient_agent=recipient_agent.name,
            run_id=None,
            parent_run_id=parent_run_id,
            model=recipient_agent.model,
            temperature=recipient_agent.temperature,
        )

        try:
            while True:  # Loop to handle tool calls
                # Create streaming response
                stream = await self.client.responses.create(
                    model=recipient_agent.model,
                    input=input_messages,
                    instructions=additional_instructions
                    or recipient_agent.instructions,
                    tools=self._prepare_tools(),
                    temperature=recipient_agent.temperature,
                    previous_response_id=self.previous_response_id,
                    tool_choice=tool_choice or "auto",
                    stream=True,
                )

                current_response_id = None
                tool_calls_in_progress = []
                current_text = ""

                async for chunk in stream:
                    # Handle response creation
                    if chunk.type == "response.created":
                        if hasattr(chunk, "id"):
                            current_response_id = chunk.id
                            self.previous_response_id = chunk.id
                        continue

                    # Handle text deltas
                    if chunk.type == "text.delta":
                        current_text += chunk.text
                        event_handler.on_text_delta(chunk.text, "")
                        response = AgentResponse(
                            type="text",
                            sender_name=recipient_agent.name,
                            receiver_name=self.agent.name,
                            content=chunk.text,
                            raw_response=chunk,
                        )
                        if self.on_response:
                            self.on_response(response)
                        yield response

                    # Handle tool calls
                    if chunk.type == "function_call":
                        event_handler.on_tool_call_created(chunk)
                        tool_calls_in_progress.append(chunk)
                        response = AgentResponse.from_tool_call(
                            chunk,
                            sender_name=recipient_agent.name,
                            receiver_name=self.agent.name,
                        )
                        if self.on_tool_call:
                            self.on_tool_call(response)
                        yield response

                        # Execute tool
                        try:
                            result = await self._execute_tool(
                                tool_call=chunk,
                                recipient_agent=recipient_agent,
                                event_handler=event_handler,
                            )
                            event_handler.on_tool_call_completed(chunk, result)

                            # Add tool call and result to input messages
                            input_messages.append(
                                {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [chunk],
                                }
                            )
                            input_messages.append(
                                {
                                    "role": "tool",
                                    "content": str(result),
                                    "tool_call_id": chunk.id,
                                }
                            )

                        except Exception as e:
                            event_handler.on_tool_call_error(chunk, str(e))
                            error_response = AgentResponse.from_error(
                                e,
                                sender_name=recipient_agent.name,
                                receiver_name=self.agent.name,
                            )
                            if self.on_error:
                                self.on_error(error_response)
                            yield error_response

                    # Handle completion
                    if chunk.type == "response.completed":
                        event_handler.on_response_completed()

                        # If we have tool calls, continue streaming with updated context
                        if tool_calls_in_progress:
                            break  # Break inner loop to start new stream with tool outputs
                        else:
                            return  # No tool calls, we're done

                # If we had tool calls, continue with a new stream
                if not tool_calls_in_progress:
                    break  # No more tool calls, exit outer loop

        except Exception as e:
            error_response = AgentResponse.from_error(
                e, sender_name=recipient_agent.name, receiver_name=self.agent.name
            )
            self.tracking_manager.track_chain_error(
                error_response, run_id, parent_run_id
            )
            if self.on_error:
                self.on_error(error_response)
            raise e

    async def _execute_tool(
        self,
        tool_call: Any,
        recipient_agent: Agent,
        event_handler: Type[AgencyEventHandler] | None,
    ) -> str:
        """Execute a tool and return its result."""
        tool = next(
            (t for t in recipient_agent.tools if t.__name__ == tool_call.function.name),
            None,
        )

        if not tool:
            raise ValueError(f"Tool {tool_call.function.name} not found")

        try:
            # Parse arguments
            args = json.loads(tool_call.function.arguments)

            # Create tool instance
            tool_instance = tool(**args)

            # Set shared state if available
            if hasattr(recipient_agent, "shared_state"):
                tool_instance._shared_state = recipient_agent.shared_state

            # Execute tool
            result = tool_instance.run()

            # Handle async results
            if inspect.iscoroutine(result):
                result = await result

            # Convert result to string
            if isinstance(result, (dict, list)):
                result = json.dumps(result)
            else:
                result = str(result)

            return result

        except Exception as e:
            print(f"Error executing tool {tool_call.function.name}: {str(e)}")
            raise

    def _check_agent_calls(self, caller_name: str, target_name: str) -> None:
        """Check if maximum number of calls between agents has been exceeded.

        Args:
            caller_name: Name of calling agent
            target_name: Name of target agent

        Raises:
            RuntimeError: If max calls exceeded
        """
        call_key = f"{caller_name}->{target_name}"
        if self._agent_call_counts[call_key] >= MAX_AGENT_CALLS:
            raise RuntimeError(
                f"Maximum number of calls ({MAX_AGENT_CALLS}) exceeded between {caller_name} and {target_name}"
            )

        self._agent_call_counts[call_key] += 1

    def _clear_agent_calls(self) -> None:
        """Clear agent call tracking state."""
        self._agent_call_counts.clear()

    async def _handle_tool_calls(
        self,
        tool_calls: list[Any],
        response: Response,
        recipient_agent: Agent,
        event_handler: Type[AgencyEventHandler] | None,
        parent_run_id: str | None,
    ) -> list[dict]:
        """Handle tool calls, including SendMessage tools which create recursive agent interactions."""
        tool_outputs = []

        # Track all tool calls at once
        self.tracking_manager.track_agent_actions(
            tool_calls, response.id, parent_run_id
        )

        # Split into sync and async tool calls
        sync_tool_calls, async_tool_calls = self._get_sync_async_tool_calls(
            tool_calls, recipient_agent
        )

        # Handle async tool calls if any
        if async_tool_calls and self.async_mode == "tools_threading":
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for tool_call in async_tool_calls:
                    futures[
                        executor.submit(
                            self._execute_tool,
                            tool_call,
                            recipient_agent,
                            event_handler,
                        )
                    ] = tool_call

                for future in as_completed(futures):
                    tool_call = futures[future]
                    try:
                        result = await future.result()
                        tool_outputs.append(
                            {
                                "role": "tool",
                                "content": str(result),
                                "tool_call_id": tool_call.id,
                            }
                        )
                        self.tracking_manager.track_tool_end(
                            output=result,
                            tool_call=tool_call,
                            parent_run_id=response.id,
                            is_retriever=tool_call.type == "file_search",
                        )
                    except Exception as e:
                        self.tracking_manager.track_tool_error(
                            error=e,
                            tool_call=tool_call,
                            parent_run_id=response.id,
                            is_retriever=tool_call.type == "file_search",
                        )
                        tool_outputs.append(
                            {
                                "role": "tool",
                                "content": f"Error: {str(e)}",
                                "tool_call_id": tool_call.id,
                            }
                        )

        # Handle sync tool calls
        sync_tool_calls += (
            async_tool_calls if not self.async_mode == "tools_threading" else []
        )
        for tool_call in sync_tool_calls:
            try:
                if tool_call.function.name.startswith("SendMessage"):
                    # Handle SendMessage tool - create new thread for recipient
                    args = json.loads(tool_call.function.arguments)
                    target_agent = self._get_recipient_agent(args["recipient"])

                    # Check agent call limits
                    try:
                        self._check_agent_calls(recipient_agent.name, target_agent.name)
                    except RuntimeError as e:
                        tool_outputs.append(
                            {
                                "role": "tool",
                                "content": f"Error: {str(e)}",
                                "tool_call_id": tool_call.id,
                            }
                        )
                        continue

                    try:
                        # Create new thread and get response
                        new_thread = Thread(
                            recipient_agent,
                            target_agent,
                            threads_callbacks=self.threads_callbacks,
                        )
                        result = await new_thread.get_response(
                            message=args["message"], parent_run_id=tool_call.id
                        )

                        tool_outputs.append(
                            {
                                "role": "tool",
                                "content": result.content,
                                "tool_call_id": tool_call.id,
                            }
                        )
                    finally:
                        # Clear agent call state after completion
                        self._clear_agent_calls()

                else:
                    # Handle other tools
                    result = await self._execute_tool(
                        tool_call=tool_call,
                        recipient_agent=recipient_agent,
                        event_handler=event_handler,
                    )
                    tool_outputs.append(
                        {
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": tool_call.id,
                        }
                    )

                # Track successful tool execution
                self.tracking_manager.track_tool_end(
                    output=result,
                    tool_call=tool_call,
                    parent_run_id=response.id,
                    is_retriever=tool_call.type == "file_search",
                )

            except Exception as e:
                # Track tool error
                self.tracking_manager.track_tool_error(
                    error=e,
                    tool_call=tool_call,
                    parent_run_id=response.id,
                    is_retriever=tool_call.type == "file_search",
                )
                tool_outputs.append(
                    {
                        "role": "tool",
                        "content": f"Error: {str(e)}",
                        "tool_call_id": tool_call.id,
                    }
                )

        # Add assistant message with tool calls
        tool_outputs.insert(
            0, {"role": "assistant", "content": None, "tool_calls": tool_calls}
        )

        return tool_outputs

    def _get_sync_async_tool_calls(
        self, tool_calls: list[Any], recipient_agent: Agent
    ) -> tuple[list[Any], list[Any]]:
        """Split tool calls into sync and async based on tool configuration."""
        async_tool_calls = []
        sync_tool_calls = []

        for tool_call in tool_calls:
            if tool_call.function.name.startswith("SendMessage"):
                sync_tool_calls.append(tool_call)
                continue

            tool = next(
                (
                    func
                    for func in recipient_agent.tools
                    if func.__name__ == tool_call.function.name
                ),
                None,
            )

            if tool is None:
                error_message = f"Tool {tool_call.function.name} not found in agent {recipient_agent.name}."
                logger.error(error_message)
                raise ValueError(error_message)

            if (
                hasattr(tool, "ToolConfig")
                and hasattr(tool.ToolConfig, "async_mode")
                and tool.ToolConfig.async_mode
            ) or self.async_mode == "tools_threading":
                async_tool_calls.append(tool_call)
            else:
                sync_tool_calls.append(tool_call)

        return sync_tool_calls, async_tool_calls

    def _get_recipient_agent(self, agent_name: str) -> Agent:
        """Get the recipient agent by name."""
        if agent_name == self.recipient_agent.name:
            return self.recipient_agent

        if hasattr(self.recipient_agent, "agency"):
            agent = self.recipient_agent.agency._get_agent_by_name(agent_name)
            if agent:
                return agent

        raise ValueError(f"Agent {agent_name} not found")

    async def _validate_response(
        self,
        recipient_agent: Agent,
        response: Response,
        validation_attempts: int = 0,
        additional_instructions: str | None = None,
        event_handler: Type[AgencyEventHandler] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Union[Response, None]:
        """Validate response using agent's validator if present."""
        if not recipient_agent.response_validator:
            return None

        try:
            recipient_agent.response_validator(message=response.output[0].text.value)
            return None
        except Exception as e:
            if validation_attempts >= recipient_agent.validation_attempts:
                return None

            # Create validation message
            content = str(e)
            try:
                evaluated_content = eval(content)
                if isinstance(evaluated_content, list):
                    content = evaluated_content
            except:
                pass

            # Get new response with validation feedback
            return await self.client.responses.create(
                instructions=additional_instructions or recipient_agent.instructions,
                model=recipient_agent.model,
                input=[{"role": "user", "content": content}],
                tools=self._prepare_tools(),
                temperature=recipient_agent.temperature,
                previous_response_id=self.previous_response_id,
                tool_choice=tool_choice or "auto",
            )

    async def _upload_file(self, file_path: str) -> str:
        """Upload a file to OpenAI and return its ID.

        Args:
            file_path: Path to file to upload

        Returns:
            File ID from OpenAI
        """
        purpose = get_file_purpose(file_path)

        with open(file_path, "rb") as f:
            file = await self.client.files.create(file=f, purpose=purpose)
            return file.id
