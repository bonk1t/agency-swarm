from __future__ import annotations

import inspect
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Type

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

    from .thread_async import ThreadAsync

logger = logging.getLogger(__name__)


class ToolNotFoundError(Exception):
    """Raised when a tool is not found in an agent's functions."""

    pass


class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails with additional context."""

    def __init__(
        self, message: str, tool_name: str, original_error: Exception, **context
    ):
        self.tool_name = tool_name
        self.original_error = original_error
        self.context = context
        super().__init__(
            f"{message} - Tool: {tool_name} - Original error: {str(original_error)}"
        )


class Thread:
    """
    A Thread represents a bidirectional communication channel between a sender and recipient.
    The recipient MUST always be an Agent, while the sender can be either a User or an Agent.

    Valid flows:
    - User -> Agent (sender=User, recipient=Agent)
    - Agent -> Agent (sender=Agent, recipient=Agent)

    Invalid flow:
    - Agent -> User (recipient can never be User)

    The Thread maintains complete conversation history locally for full state control.
    Every message sent through get_response() gets a response from the recipient Agent.

    This class handles both synchronous and parallel tool execution through threading.
    Note that while tools themselves are synchronous, they can be executed in parallel
    using thread pools when configured with ToolConfig.async_mode = "threading".

    This is different from ThreadAsync which handles non-blocking agent communication,
    not asynchronous tool execution.
    """

    def __init__(
        self,
        sender: Agent | User,
        recipient: Agent,
        messages: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize a Thread instance.

        Args:
            sender: The initiator of the conversation (can be User or Agent)
            recipient: The Agent that will receive and respond to messages
            messages: Initial message history for the thread

        Raises:
            TypeError: If recipient is not an Agent
        """
        if not isinstance(recipient, Agent):
            raise TypeError("Thread recipient must be an Agent")

        self.sender = sender
        self.recipient = recipient
        self.messages = messages or []
        self.id = str(uuid.uuid4())
        self._client = None
        self.shared_state = None
        self.tracking_manager = TrackingManager()
        self._agent_call_counts = {}

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if not self._client:
            self._client = get_openai_client()
        return self._client

    def _prepare_tools(self) -> list[dict]:
        """Convert agent tools to OpenAI format for AsyncResponses API."""
        if not self.recipient.tools:
            return []

        tools = []
        for tool in self.recipient.tools:
            schema = tool.openai_schema
            tools.append(schema)
        return tools

    def _format_message(
        self, message: str, message_files: list[str] | None = None
    ) -> list[dict]:
        """Format a message into OpenAI's expected message format for AsyncResponses API.

        Args:
            message: String message to format
            message_files: Optional list of file IDs to attach

        Returns:
            List of properly formatted message dicts for AsyncResponses API
        """
        if not isinstance(message, str):
            raise ValueError("Message must be a string")

        content = [{"type": "input_text", "text": message}]
        if message_files:
            for file_id in message_files:
                content.append({"type": "input_file", "file_id": file_id})

        return [{"role": "user", "content": content}]

    async def _prepare_input(
        self, message: str, message_files: list[str] | None = None
    ) -> list[dict]:
        """Prepare input messages including conversation history.

        Formats new message and combines it with existing conversation history.

        Args:
            message: New message to add to history
            message_files: Optional list of file IDs to attach

        Returns:
            List of messages including history
        """
        # Initialize messages list if None
        if self.messages is None:
            self.messages = []

        # Format new message consistently
        new_messages = self._format_message(message, message_files)

        # Add new message to history
        self.messages.extend(new_messages)

        # Return copy to avoid modifying history
        return self.messages.copy()

    def _extract_text_content(self, response: Response) -> str:
        """Extract text content from a response in a consistent way.

        Args:
            response: The OpenAI response object

        Returns:
            Text content from the response, or empty string if none found
        """
        content = ""
        for output_item in response.output:
            if output_item.type == "message" and output_item.content:
                content = output_item.content[0].text
                break
        return content

    async def _prepare_conversation_input(
        self,
        message: str,
        message_files: list[str] | None = None,
    ) -> list[dict]:
        """Prepare conversation input for the API call.

        Always sends the full conversation history to maintain context properly.
        State is maintained through the complete message history.

        Args:
            message: The new message to send
            message_files: Optional list of file IDs to attach

        Returns:
            List of message dictionaries formatted for the API
        """
        # Format the new message
        new_message = self._format_message(message, message_files)

        # Initialize messages list if needed
        self.messages = self.messages or []

        # Add new message to history
        self.messages.extend(new_message)

        # Return full conversation history
        return self.messages.copy()

    async def get_response(
        self,
        message: str,
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ) -> AgentResponse:
        """Get a response from the recipient Agent.

        Every message sent through this method gets a response from the recipient Agent.
        The recipient is always an Agent (never a User).

        Args:
            message: The message to send
            message_files: Optional list of file IDs to attach
            recipient_agent: Optional override for the recipient Agent
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice
            parent_run_id: Optional parent run ID for tracking

        Returns:
            AgentResponse containing the Agent's response
        """
        if not recipient_agent:
            recipient_agent = self.recipient

        # Set correct sender/receiver names based on message flow
        sender_name = "user" if isinstance(self.sender, User) else self.sender.name
        receiver_name = recipient_agent.name

        # Debug info with clear message flow
        logger.debug(f"RESPONSE:[ {sender_name} -> {recipient_agent.name} ]")

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
            # Prepare input with conversation history
            messages = await self._prepare_conversation_input(message, message_files)

            # Create response with complete history
            response: Response = await self.client.responses.create(
                model=recipient_agent.model,
                input=messages,  # Use full history
                instructions=additional_instructions or recipient_agent.instructions,
                tools=self._prepare_tools() if recipient_agent.tools else None,
                temperature=recipient_agent.temperature
                if recipient_agent.temperature is not None
                else 0.7,
                tool_choice=tool_choice if tool_choice is not None else "auto",
                stream=False,
            )

            # Process response output
            tool_calls = []
            assistant_message = None

            for output_item in response.output:
                if output_item.type == "function_call":
                    tool_calls.append(
                        {
                            "type": "function_call",
                            "id": output_item.id,
                            "name": output_item.name,
                            "arguments": output_item.arguments,
                        }
                    )
                elif output_item.type == "message" and output_item.content:
                    assistant_message = output_item.content[0].text

            # Handle tool calls if present
            if tool_calls:
                tool_outputs = await self._handle_tool_calls(
                    tool_calls=tool_calls,
                    response=response,
                    recipient_agent=recipient_agent,
                    event_handler=None,
                    parent_run_id=parent_run_id,
                )

                # Add function calls to history
                self.messages.extend(
                    [
                        *[{"type": "function_call", **call} for call in tool_calls],
                        *[
                            {
                                "type": "function_call_output",
                                "call_id": output["call_id"],
                                "output": output["content"][0]["text"]
                                if isinstance(output["content"], list)
                                else output["content"],
                            }
                            for output in tool_outputs
                            if output["role"] == "tool"
                        ],
                    ]
                )

                # Create agent response with tool outputs
                agent_response = AgentResponse(
                    type="tool_call",
                    sender_name=sender_name,
                    receiver_name=receiver_name,
                    content=tool_outputs,
                    raw_response=response,
                )
            else:
                # Create text response
                agent_response = AgentResponse(
                    type="text",
                    sender_name=sender_name,
                    receiver_name=receiver_name,
                    content=assistant_message,
                    raw_response=response,
                )

                # Add assistant's message to history
                self.messages.append(
                    {
                        "type": "message",
                        "content": [{"type": "text", "text": assistant_message}],
                    }
                )

            # Track successful completion
            self.tracking_manager.end_run(agent_response, run_id, parent_run_id)

            return agent_response

        except Exception as e:
            # Track error and re-raise
            self.tracking_manager.track_chain_error(e, run_id, parent_run_id)
            raise

    async def get_response_stream(
        self,
        message: str,
        event_handler: Type[AgencyEventHandler],
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        parent_run_id: str | None = None,
    ) -> AsyncIterator[AgentResponse]:
        """Streaming version of get_response that yields AgentResponse events.

        Every message sent through this method gets a streaming response from the recipient Agent.
        The recipient is always an Agent (never a User).

        Args:
            message: The message to send to the Agent
            event_handler: Handler class for streaming events
            message_files: Optional list of file IDs to attach
            recipient_agent: Optional override for the recipient Agent
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice for specifying which tools to use
            parent_run_id: Optional parent run ID for tracking

        Yields:
            AgentResponse events for streaming responses

        Note:
            Full conversation history is maintained locally and sent with each request
            to ensure proper state management and context preservation.

        Raises:
            TypeError: If message is not a string
        """
        if not isinstance(message, str):
            raise TypeError("Message must be a string")

        if not recipient_agent:
            recipient_agent = self.recipient

        # Debug info
        sender_name = "user" if isinstance(self.sender, User) else self.sender.name
        logger.debug(f"RESPONSE_STREAM:[ {sender_name} -> {recipient_agent.name} ]")

        # Prepare input messages using centralized helper
        input_messages = await self._prepare_conversation_input(message, message_files)

        # Track start of interaction
        run_id = self.tracking_manager.start_run(
            message=message,
            sender_agent=self.sender.name,
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
                    tools=self._prepare_tools() if recipient_agent.tools else None,
                    temperature=recipient_agent.temperature
                    if recipient_agent.temperature is not None
                    else 0.0,
                    tool_choice=tool_choice or "auto",
                    stream=True,
                )

                tool_calls_in_progress = []

                async for chunk in stream:
                    # Handle response creation
                    if chunk.type == "response.created":
                        continue

                    # Handle text deltas
                    if chunk.type == "text.delta":
                        event_handler.on_text_delta(chunk.text, "")
                        response = AgentResponse(
                            type="text",
                            sender_name=sender_name,
                            receiver_name=recipient_agent.name,
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
                            sender_name=sender_name,
                            receiver_name=recipient_agent.name,
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

                            # Add tool response to history
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "content": result,
                                    "tool_call_id": chunk.id,
                                }
                            )

                        except Exception as e:
                            event_handler.on_tool_call_error(chunk, str(e))
                            error_response = AgentResponse.from_error(
                                e,
                                sender_name=sender_name,
                                receiver_name=recipient_agent.name,
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
                e, sender_name=sender_name, receiver_name=recipient_agent.name
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
        """Execute a tool and return its result.

        This method handles three types of tool execution:
        1. Async execution via run_async() - for tools configured with async_mode
        2. Coroutine execution - for tools that return coroutines from run()
        3. Synchronous execution - for regular tools

        The execution order is:
        1. Try run_async() if tool is configured for async
        2. Try run() and await if it returns a coroutine
        3. Fall back to synchronous run()
        """
        tool = next(
            (t for t in recipient_agent.tools if t.__name__ == tool_call.function.name),
            None,
        )

        if not tool:
            error_msg = f"Tool {tool_call.function.name} not found"
            logger.error(error_msg)
            raise ToolNotFoundError(error_msg)

        try:
            # Parse arguments
            args = json.loads(tool_call.function.arguments)

            # Log tool execution with context
            logger.debug(
                f"Executing tool {tool_call.function.name} with args: {json.dumps(args, indent=2)}"
            )

            # Create tool instance
            tool_instance = tool(**args)

            # Set shared state if available
            if hasattr(recipient_agent, "shared_state"):
                tool_instance._shared_state = recipient_agent.shared_state

            # Try to execute the tool in the appropriate mode
            if (
                hasattr(tool, "ToolConfig")
                and hasattr(tool.ToolConfig, "async_mode")
                and tool.ToolConfig.async_mode
            ):
                # Use run_async for tools configured for async execution
                result = await tool_instance.run_async()
            else:
                # Try regular run() first
                result = tool_instance.run()

                # If run() returned a coroutine, await it
                if inspect.iscoroutine(result):
                    result = await result

            # Convert result to string
            if isinstance(result, (dict, list)):
                result = json.dumps(result)
            else:
                result = str(result)

            logger.debug(
                f"Tool {tool_call.function.name} executed successfully with result type: {type(result)}"
            )
            return result

        except Exception as e:
            error_context = {
                "tool_arguments": tool_call.function.arguments,
                "recipient_agent": recipient_agent.name,
                "event_handler_type": type(event_handler).__name__
                if event_handler
                else None,
            }

            logger.error(
                f"Error executing tool {tool_call.function.name}: {str(e)}\n"
                f"Context: {json.dumps(error_context, indent=2)}"
            )

            raise ToolExecutionError(
                message="Failed to execute tool",
                tool_name=tool_call.function.name,
                original_error=e,
                **error_context,
            )

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
        tool_calls: list[dict],
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

        # If this is an async thread, update tool call tracking
        if self.__class__.__name__ == "ThreadAsync":
            self._tool_calls_in_progress = tool_calls.copy()
            self._tool_results.clear()

        # Split into synchronous and parallel execution groups
        sync_tool_calls, parallel_tool_calls = self._get_sync_async_tool_calls(
            tool_calls, recipient_agent
        )

        # Handle synchronous tool calls
        sync_tool_calls += (
            parallel_tool_calls
            if getattr(self, "async_mode", "none") != "tools_threading"
            else []
        )

        for tool_call in sync_tool_calls:
            try:
                tool_name = tool_call.get("function", {}).get("name")

                if not tool_name:
                    error_message = f"Invalid tool call format: {tool_call}"
                    logger.error(error_message)
                    raise ValueError(error_message)

                if tool_name.startswith("SendMessage"):
                    # Handle SendMessage tool - create new thread for recipient
                    args = json.loads(tool_call["function"]["arguments"])
                    target_agent = self._get_recipient_agent(args["recipient"])

                    # Check agent call limits
                    try:
                        self._check_agent_calls(recipient_agent.name, target_agent.name)
                    except RuntimeError as e:
                        if self.__class__.__name__ == "ThreadAsync":
                            self._tool_results[tool_call["id"]] = str(e)
                            if tool_call in self._tool_calls_in_progress:
                                self._tool_calls_in_progress.remove(tool_call)
                        tool_outputs.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": tool_call["id"],
                                "content": [
                                    {"type": "output_text", "text": f"Error: {str(e)}"}
                                ],
                            }
                        )
                        continue

                    try:
                        # Create new thread and get response
                        new_thread = Thread(
                            recipient_agent,
                            target_agent,
                            # Forward async settings for consistency
                            async_mode=getattr(self, "async_mode", "none"),
                            max_workers=getattr(self, "max_workers", 5),
                        )
                        result = await new_thread.get_response(
                            message=args["message"], parent_run_id=tool_call["id"]
                        )

                        if self.__class__.__name__ == "ThreadAsync":
                            self._tool_results[tool_call["id"]] = result.content
                            if tool_call in self._tool_calls_in_progress:
                                self._tool_calls_in_progress.remove(tool_call)

                        tool_outputs.append(
                            {
                                "role": "tool",
                                "type": "function_call_output",
                                "call_id": tool_call["id"],
                                "content": [
                                    {"type": "output_text", "text": result.content}
                                ],
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

                    if self.__class__.__name__ == "ThreadAsync":
                        self._tool_results[tool_call["id"]] = result
                        if tool_call in self._tool_calls_in_progress:
                            self._tool_calls_in_progress.remove(tool_call)

                    tool_outputs.append(
                        {
                            "role": "tool",
                            "type": "function_call_output",
                            "call_id": tool_call["id"],
                            "content": [{"type": "output_text", "text": str(result)}],
                        }
                    )

                # Track successful tool execution
                self.tracking_manager.track_tool_end(
                    output=result,
                    tool_call=tool_call,
                    parent_run_id=response.id,
                )

            except Exception as e:
                # Track tool error
                if self.__class__.__name__ == "ThreadAsync":
                    self._tool_results[tool_call["id"]] = str(e)
                    if tool_call in self._tool_calls_in_progress:
                        self._tool_calls_in_progress.remove(tool_call)

                tool_outputs.append(
                    {
                        "role": "tool",
                        "type": "function_call_output",
                        "call_id": tool_call["id"],
                        "content": [
                            {"type": "output_text", "text": f"Error: {str(e)}"}
                        ],
                    }
                )

                self.tracking_manager.track_tool_error(
                    error=e,
                    tool_call=tool_call,
                    parent_run_id=response.id,
                )
                tool_outputs.append(
                    {
                        "role": "tool",
                        "content": f"Error: {str(e)}",
                        "tool_call_id": tool_call.id,
                    }
                )

        # Handle parallel tool execution if any tools are configured for it
        if (
            parallel_tool_calls
            and getattr(self, "async_mode", "none") == "tools_threading"
        ):
            import asyncio

            # Execute all parallel tools concurrently using their run_async methods
            parallel_results = await asyncio.gather(
                *[
                    self._execute_tool(tool_call, recipient_agent, event_handler)
                    for tool_call in parallel_tool_calls
                ],
                return_exceptions=False,
            )

            # Process results
            for tool_call, result in zip(parallel_tool_calls, parallel_results):
                try:
                    tool_output = {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    }

                    if isinstance(self, ThreadAsync):
                        self._tool_results[tool_call.id] = result
                        if tool_call in self._tool_calls_in_progress:
                            self._tool_calls_in_progress.remove(tool_call)

                    self.tracking_manager.track_tool_end(
                        output=result,
                        tool_call=tool_call,
                        parent_run_id=response.id,
                    )

                    tool_outputs.append(tool_output)
                except Exception as e:
                    # Handle errors
                    if isinstance(self, ThreadAsync):
                        self._tool_results[tool_call.id] = str(e)
                        if tool_call in self._tool_calls_in_progress:
                            self._tool_calls_in_progress.remove(tool_call)

                    self.tracking_manager.track_tool_error(
                        error=e,
                        tool_call=tool_call,
                        parent_run_id=response.id,
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
        self, tool_calls: list[dict], recipient_agent: Agent
    ) -> tuple[list[dict], list[dict]]:
        """Split tool calls into synchronous and parallel (threaded) execution groups.

        Tools can be executed in two ways:
        1. Synchronously - one after another in the main thread
        2. In parallel - using a thread pool when marked with ToolConfig.async_mode = "threading"

        Note: The tools themselves are synchronous Python functions, but can be run in separate
        threads for concurrent execution when configured for parallel execution.

        Args:
            tool_calls: List of tool calls to process
            recipient_agent: Agent whose tools are being called

        Returns:
            Tuple of (sync_tool_calls, parallel_tool_calls)

        Raises:
            ValueError: If a tool is not found in the agent's tools
        """
        parallel_tool_calls: list[dict] = []
        sync_tool_calls: list[dict] = []

        for tool_call in tool_calls:
            # Handle both object and dictionary access
            tool_name = tool_call.get("function", {}).get("name")

            if not tool_name:
                error_message = f"Invalid tool call format: {tool_call}"
                logger.error(error_message)
                raise ValueError(error_message)

            if tool_name.startswith("SendMessage"):
                sync_tool_calls.append(tool_call)
                continue

            tool = next(
                (func for func in recipient_agent.tools if func.__name__ == tool_name),
                None,
            )

            if tool is None:
                error_message = (
                    f"Tool {tool_name} not found in agent {recipient_agent.name}."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            # Check if tool is configured for parallel execution
            if (
                hasattr(tool, "ToolConfig")
                and hasattr(tool.ToolConfig, "async_mode")
                and tool.ToolConfig.async_mode
            ) or self.async_mode == "tools_threading":
                parallel_tool_calls.append(tool_call)
            else:
                sync_tool_calls.append(tool_call)

        return sync_tool_calls, parallel_tool_calls

    def _get_recipient_agent(self, agent_name: str) -> Agent:
        """Get the recipient agent by name."""
        if agent_name == self.recipient.name:
            return self.recipient

        if hasattr(self.recipient, "agency"):
            agent = self.recipient.agency._get_agent_by_name(agent_name)
            if agent:
                return agent

        raise ValueError(f"Agent {agent_name} not found")

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
