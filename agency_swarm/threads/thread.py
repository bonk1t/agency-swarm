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
    async_mode: str = None
    max_workers: int = 4

    @property
    def thread_url(self):
        return f"https://platform.openai.com/playground/assistants?assistant={self.recipient_agent.id}&mode=assistant&thread={self.id}"

    @property
    def thread(self):
        self.init_thread()

        if not self._thread:
            logger.debug(f"Retrieving thread {self.id}")
            self._thread = self.client.beta.threads.retrieve(self.id)

        return self._thread

    def __init__(self, agent: Union[Agent, User], recipient_agent: Agent):
        self.agent = agent
        self.recipient_agent = recipient_agent

        self.client = get_openai_client()

        self.id = None
        self._thread = None
        self._run = None
        self._stream = None

        self._num_run_retries = 0
        # names of recipient agents that were called in SendMessage tool
        # needed to prevent agents calling the same recipient agent multiple times
        self._called_recepients = []

        self.terminal_states = [
            "cancelled",
            "completed",
            "failed",
            "expired",
            "incomplete",
        ]

        self._tracking_manager = TrackingManager()

    def init_thread(self):
        self._called_recepients = []
        self._num_run_retries = 0

        if self.id:
            return

        self._thread = self.client.beta.threads.create()
        self.id = self._thread.id
        if self.recipient_agent.examples:
            for example in self.recipient_agent.examples:
                self.client.beta.threads.messages.create(
                    thread_id=self.id,
                    **example,
                )

    def get_completion_stream(
        self,
        message: str | list[dict] | None,
        event_handler: Type[AgencyEventHandler],
        message_files: list[str] | None = None,
        attachments: list[Attachment] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: AssistantToolChoice | None = None,
        response_format: dict | None = None,
        parent_run_id: str | None = None,
    ) -> Generator[MessageOutput, None, str]:
        return self.get_completion(
            message,
            message_files,
            attachments,
            recipient_agent,
            additional_instructions,
            event_handler,
            tool_choice,
            yield_messages=False,
            response_format=response_format,
            parent_run_id=parent_run_id,
        )

    def get_completion(
        self,
        message: str | list[dict] | None,
        message_files: list[str] | None = None,
        attachments: list[Attachment] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        event_handler: Type[AgencyEventHandler] | None = None,
        tool_choice: AssistantToolChoice | None = None,
        yield_messages: bool = False,
        response_format: dict | None = None,
        parent_run_id: str | None = None,
    ) -> Generator[MessageOutput, None, str]:
        """
        Primary entry point for sending messages to the recipient agent and handling
        the completion (including tool calls, validations, and re-tries).
        """

        # 1. Prepare basic thread and attachments
        self.init_thread()
        if not recipient_agent:
            recipient_agent = self.recipient_agent
        attachments = self._setup_attachments(
            attachments, message_files, recipient_agent
        )

        # 2. Optionally set the event handler's agent references
        if event_handler:
            event_handler.set_agent(self.agent)
            event_handler.set_recipient_agent(recipient_agent)

        # 3. Print debug info and send user message
        self._debug_print_sender_and_url(recipient_agent)
        message_obj = None
        if message:
            message_obj = self.create_message(
                message=message, role="user", attachments=attachments
            )
            if yield_messages:
                yield MessageOutput(
                    "text", self.agent.name, recipient_agent.name, message, message_obj
                )

        # 4. Create run (conversation block)
        self._create_run(
            recipient_agent,
            additional_instructions,
            event_handler,
            tool_choice,
            response_format=response_format,
        )
        final_output = None

        # 5. Fire run start callbacks
        self._tracking_manager.start_run(
            message,
            self.agent.name,
            recipient_agent.name,
            run_id=self._run.id,
            parent_run_id=parent_run_id,
            message_obj=message_obj,
            model=self._run.model,
            temperature=self._run.temperature,
        )

        # 6. Main try/except around the run loop
        final_output = yield from self._execute_main_loop(
            yield_messages=yield_messages,
            recipient_agent=recipient_agent,
            event_handler=event_handler,
            parent_run_id=parent_run_id,
            additional_instructions=additional_instructions,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        if final_output is None:
            raise Exception("No output was generated from the execution loop")

        return final_output

    def _execute_main_loop(
        self,
        yield_messages: bool,
        recipient_agent: Agent,
        event_handler: Type[AgencyEventHandler] | None,
        parent_run_id: str | None,
        additional_instructions: str | None,
        tool_choice: AssistantToolChoice | None,
        response_format: dict | None,
    ) -> Generator[MessageOutput, None, str]:
        """
        Encapsulates the 'while True' run loop from get_completion to reduce
        cognitive load in the main method. Yields any MessageOutput events
        and returns the final output string.
        """
        error_attempts = 0
        validation_attempts = 0
        full_message = ""
        final_output = None

        while True:
            self._run_until_done()

            if self._run.status == "requires_action":
                maybe_output = yield from self._handle_run_requires_action(
                    recipient_agent,
                    event_handler,
                    yield_messages,
                    parent_run_id,
                    additional_instructions,
                )
                if maybe_output is not None:
                    final_output = maybe_output
                    break

            elif self._run.status == "failed":
                # If the run fails, try re-running on certain error messages
                full_message += self._get_last_message_text()
                retry_successful = self._try_run_failed_recovery(
                    error_attempts,
                    recipient_agent,
                    additional_instructions,
                    event_handler,
                    tool_choice,
                    response_format,
                    parent_run_id,
                )
                error_attempts += 1
                if not retry_successful:
                    raise Exception(
                        "OpenAI Run Failed. Error: ", self._run.last_error.message
                    )

            elif self._run.status == "incomplete":
                self._on_run_incomplete(parent_run_id)

            else:
                # final assistant message
                message_obj = self._get_last_assistant_message()
                last_message = message_obj.content[0].text.value
                full_message += last_message

                if yield_messages:
                    yield MessageOutput(
                        "text",
                        recipient_agent.name,
                        self.agent.name,
                        last_message,
                        message_obj,
                    )

                result = self._validate_assistant_response(
                    recipient_agent,
                    last_message,
                    validation_attempts,
                    yield_messages,
                    additional_instructions,
                    event_handler,
                    tool_choice,
                    response_format,
                )
                if result is not None:
                    # The function no longer yields, so `result` is a dict, not a generator
                    for mo in result.get("message_outputs", []):
                        yield mo  # yield the stored MessageOutput objects
                    validation_attempts = result["validation_attempts"]
                    if result["continue_loop"]:
                        continue

                if final_output is None:
                    final_output = last_message
                break

        return final_output

    def _create_run(
        self,
        message: str,
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        event_handler: Type[AgencyEventHandler] | None = None,
        tool_choice: AssistantToolChoice | None = None,
        temperature: float | None = None,
        response_format: dict | None = None,
    ):
        # Always start from a clean slate
        self._ensure_no_active_run(action="cancel")
        try:
            if event_handler:
                with self.client.beta.threads.runs.stream(
                    thread_id=self.id,
                    event_handler=event_handler(),
                    assistant_id=recipient_agent.id,
                    additional_instructions=additional_instructions,
                    tool_choice=tool_choice,
                    max_prompt_tokens=recipient_agent.max_prompt_tokens,
                    max_completion_tokens=recipient_agent.max_completion_tokens,
                    truncation_strategy=recipient_agent.truncation_strategy,
                    temperature=temperature,
                    extra_body={
                        "parallel_tool_calls": recipient_agent.parallel_tool_calls
                    },
                    response_format=response_format,
                ) as stream:
                    stream.until_done()
                    self._run = stream.get_final_run()
            else:
                self._run = self.client.beta.threads.runs.create(
                    thread_id=self.id,
                    assistant_id=recipient_agent.id,
                    additional_instructions=additional_instructions,
                    tool_choice=tool_choice,
                    max_prompt_tokens=recipient_agent.max_prompt_tokens,
                    max_completion_tokens=recipient_agent.max_completion_tokens,
                    truncation_strategy=recipient_agent.truncation_strategy,
                    temperature=temperature,
                    parallel_tool_calls=recipient_agent.parallel_tool_calls,
                    response_format=response_format,
                )
                self._run = self.client.beta.threads.runs.poll(
                    thread_id=self.id,
                    run_id=self._run.id,
                )
        except APIError as e:
            match = re.search(
                r"Thread (\w+) already has an active run (\w+)", e.message
            )
            if match:
                self.cancel_run(
                    thread_id=match.groups()[0],
                    run_id=match.groups()[1],
                    check_status=False,
                )
                # Reattempt creating a new run after cancellation.
                return self._create_run(
                    recipient_agent,
                    additional_instructions,
                    event_handler,
                    tool_choice,
                    temperature=temperature,
                    response_format=response_format,
                )
            elif (
                "The server had an error processing your request" in e.message
                and self._num_run_retries < 3
            ):
                time.sleep(1)
                self._num_run_retries += 1
                return self._create_run(
                    recipient_agent,
                    additional_instructions,
                    event_handler,
                    tool_choice,
                    temperature=temperature,
                    response_format=response_format,
                )
            else:
                raise e

    def _run_until_done(self):
        while self._run.status in ["queued", "in_progress", "cancelling"]:
            time.sleep(0.5)
            self._run = self.client.beta.threads.runs.retrieve(
                thread_id=self.id, run_id=self._run.id
            )

    def submit_tool_outputs(self, tool_outputs, event_handler=None, poll=True):
        if not poll:
            self._run = self.client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.id, run_id=self._run.id, tool_outputs=tool_outputs
            )
        else:
            if not event_handler:
                self._run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=self.id, run_id=self._run.id, tool_outputs=tool_outputs
                )
            else:
                with self.client.beta.threads.runs.submit_tool_outputs_stream(
                    thread_id=self.id,
                    run_id=self._run.id,
                    tool_outputs=tool_outputs,
                    event_handler=event_handler(),
                ) as stream:
                    stream.until_done()
                    self._run = stream.get_final_run()

    def cancel_run(self, thread_id=None, run_id=None, check_status=True):
        if (
            check_status
            and (not self._run or self._run.status in self.terminal_states)
            and not run_id
        ):
            return

        try:
            actual_thread_id = thread_id or self.id
            actual_run_id = run_id or (self._run.id if self._run else None)

            if not actual_run_id:
                logger.warning(
                    f"Can't cancel without a run ID: thread_id={actual_thread_id}"
                )
                return

            self._run = self.client.beta.threads.runs.cancel(
                thread_id=actual_thread_id, run_id=actual_run_id
            )

            self._run = self.client.beta.threads.runs.poll(
                thread_id=actual_thread_id,
                run_id=actual_run_id,
                poll_interval_ms=500,
            )
        except BadRequestError as e:
            if "Cannot cancel run with status" in e.message:
                self._run = self.client.beta.threads.runs.poll(
                    thread_id=actual_thread_id,
                    run_id=actual_run_id,
                    poll_interval_ms=500,
                )
            else:
                raise e

    def _get_last_message_text(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.id, limit=1)

        if len(messages.data) == 0 or len(messages.data[0].content) == 0:
            return ""

        return messages.data[0].content[0].text.value

    def _get_last_assistant_message(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.id, limit=1)

        if len(messages.data) == 0 or len(messages.data[0].content) == 0:
            raise Exception("No messages found in the thread")

        message = messages.data[0]

        if message.role == "assistant":
            return message

        raise Exception("No assistant message found in the thread")

    def create_message(
        self,
        message: str | list[dict],
        role: str = "user",
        attachments: list[Attachment] | None = None,
    ) -> Message:
        # Never post while a run is still alive
        self._ensure_no_active_run(action="wait")
        try:
            return self.client.beta.threads.messages.create(
                thread_id=self.id, role=role, content=message, attachments=attachments
            )
        except BadRequestError as e:
            regex = re.compile(
                r"Can't add messages to thread_([a-zA-Z0-9]+) while a run run_([a-zA-Z0-9]+) is active\."
            )
            match = regex.search(str(e))

            if match:
                thread_id, run_id = match.groups()
                thread_id = f"thread_{thread_id}"
                run_id = f"run_{run_id}"

                self.cancel_run(thread_id=thread_id, run_id=run_id)

                return self.client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role=role,
                    content=message,
                    attachments=attachments,
                )
            else:
                raise e

    def execute_tool(
        self,
        tool_call: ToolCall,
        recipient_agent=None,
        event_handler=None,
        tool_outputs_and_names=None,
    ) -> tuple[str | Generator[MessageOutput, None, None], bool]:
        if not recipient_agent:
            recipient_agent = self.recipient_agent
        if tool_outputs_and_names is None:
            tool_outputs_and_names = []

        is_retriever = tool_call.type == "file_search"

        tool_name = tool_call.function.name
        funcs = recipient_agent.functions
        tool = next((func for func in funcs if func.__name__ == tool_name), None)

        try:
            # Track start of tool execution
            self._tracking_manager.track_tool_start(
                tool_call=tool_call,
                run=self._run,
                agent_name=self.agent.name,
                recipient_agent_name=recipient_agent.name,
                is_retriever=is_retriever,
            )

            # init tool
            args = tool_call.function.arguments
            args = json.loads(args) if args else {}
            tool_instance = tool(**args)

            # check if the tool is already called
            for existing_tool_name in [name for name, _ in tool_outputs_and_names]:
                if tool_name == existing_tool_name and (
                    hasattr(tool_instance, "ToolConfig")
                    and hasattr(tool_instance.ToolConfig, "one_call_at_a_time")
                    and tool_instance.ToolConfig.one_call_at_a_time
                ):
                    error_message = f"Error: Function {tool_name} is already called. You can only call this function once at a time. Please wait for the previous call to finish before calling it again."
                    raise RuntimeError(error_message)

            # for send message tools, don't allow calling the same recipient agent multiple times
            if tool_name.startswith("SendMessage"):
                if tool_instance.recipient.value in self._called_recepients:
                    error_message = f"Error: Agent {tool_instance.recipient.value} has already been called. You can only call each agent once at a time. Please wait for the previous call to finish before calling it again."
                    raise RuntimeError(error_message)

                self._called_recepients.append(tool_instance.recipient.value)

            tool_instance._caller_agent = recipient_agent
            tool_instance._event_handler = event_handler
            tool_instance._tool_call = tool_call

            output = tool_instance.run()
            return output, tool_instance.ToolConfig.output_as_result

        except Exception as e:
            error_message = f"Error: {e}"
            if "For further information visit" in error_message:
                error_message = error_message.split("For further information visit")[0]

            # Track error
            self._tracking_manager.track_tool_error(
                error=Exception(error_message),
                tool_call=tool_call,
                parent_run_id=self._run.id,
                is_retriever=is_retriever,
            )

            return error_message, False

    def _await_coroutines(self, tool_outputs):
        async_tool_calls = []
        for tool_output in tool_outputs:
            if inspect.iscoroutine(tool_output["output"]):
                async_tool_calls.append(tool_output)

        if async_tool_calls:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop = asyncio.get_event_loop()

            results = loop.run_until_complete(
                asyncio.gather(
                    *[call["output"] for call in async_tool_calls],
                    return_exceptions=True,  # Capture exceptions to set as individual results
                )
            )

            for tool_output, result in zip(async_tool_calls, results):
                tool_output["output"] = str(result)

        return tool_outputs

    def _get_sync_async_tool_calls(
        self, tool_calls: list[RequiredActionFunctionToolCall], recipient_agent: Agent
    ):
        async_tool_calls = []
        sync_tool_calls = []

        for tool_call in tool_calls:
            try:
                if tool_call.function.name.startswith("SendMessage"):
                    sync_tool_calls.append(tool_call)
                    continue

                tool = next(
                    (
                        func
                        for func in recipient_agent.functions
                        if func.__name__ == tool_call.function.name
                    ),
                    None,
                )

                # Check if the tool found has async_mode = 'threading'
                if (
                    hasattr(tool, "ToolConfig")
                    and getattr(tool.ToolConfig, "async_mode", None) == "threading"
                ):
                    async_tool_calls.append(tool_call)
                else:
                    sync_tool_calls.append(tool_call)

            except Exception as e:
                error_context = {
                    "tool_arguments": tool_call.function.arguments,
                    "recipient_agent": recipient_agent.name,
                    "event_handler_type": type(event_handler).__name__
                    if event_handler
                    else None,
                }

                logger.error(
                    f"Error processing tool {tool_call.function.name}: {str(e)}\n"
                    f"Context: {json.dumps(error_context, indent=2)}"
                )

                # Re-raise as a more specific error, possibly add to outputs later if needed
                raise ToolExecutionError(
                    message="Failed to process tool call within _get_sync_async_tool_calls",
                    tool_name=tool_call.function.name,
                    original_error=e,
                    **error_context,
                )

        return sync_tool_calls, async_tool_calls

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
                    except Exception as e:
                        error_context = {
                            "tool_arguments": tool_call.function.arguments,
                            "recipient_agent": recipient_agent.name,
                            "event_handler_type": type(event_handler).__name__
                            if event_handler
                            else None,
                        }

                        logger.error(
                            f"Error processing tool {tool_call.function.name}: {str(e)}\n"
                            f"Context: {json.dumps(error_context, indent=2)}"
                        )

                        # Re-raise as a more specific error, possibly add to outputs later if needed
                        raise ToolExecutionError(
                            message="Failed to process tool call within _handle_tool_calls",
                            tool_name=tool_call.function.name,
                            original_error=e,
                            **error_context,
                        )

                    # Process futures
                    try:
                        async for future in as_completed(futures):
                            tool_call = futures[future]
                            output, output_as_result = await future
                            if output_as_result:
                                self.cancel_run()
                                final_output = output
                                break
                    except Exception as e:
                        logger.error(f"Error executing async tool call: {e}")
                        output = f"Error: {str(e)}"
                        output_as_result = False

            except Exception as e:
                error_context = {
                    "tool_arguments": tool_call.function.arguments,
                    "recipient_agent": recipient_agent.name,
                    "event_handler_type": type(event_handler).__name__
                    if event_handler
                    else None,
                }

                logger.error(
                    f"Error processing tool {tool_call.function.name}: {str(e)}\n"
                    f"Context: {json.dumps(error_context, indent=2)}"
                )

                # Re-raise as a more specific error, possibly add to outputs later if needed
                raise ToolExecutionError(
                    message="Failed to process tool call within _handle_tool_calls",
                    tool_name=tool_call.function.name,
                    original_error=e,
                    **error_context,
                )

        # If a tool call had "output_as_result", return immediately
        if final_output is not None:
            return final_output

        tool_outputs = [t for _, t in tool_outputs_and_names]
        tool_names = [n for n, _ in tool_outputs_and_names]

        tool_outputs = self._await_coroutines(tool_outputs)

        for to_ in tool_outputs:
            if not isinstance(to_["output"], str):
                to_["output"] = str(to_["output"])

        if event_handler:
            event_handler.set_agent(self.agent)
            event_handler.set_recipient_agent(recipient_agent)

        try:
            self.submit_tool_outputs(tool_outputs, event_handler)
        except BadRequestError as e:
            if 'Runs in status "expired"' in e.message:
                self.create_message(
                    message="Previous request timed out. Please repeat the exact same tool calls in the exact same order with the same arguments.",
                    role="user",
                )
                self._create_run(
                    recipient_agent,
                    additional_instructions,
                    event_handler,
                    "required",
                    temperature=0,
                )
                self._run_until_done()

                if self._run.status != "requires_action":
                    raise Exception(
                        "Run Failed. Error: ",
                        self._run.last_error or self._run.incomplete_details,
                    )
                tool_calls = self._run.required_action.submit_tool_outputs.tool_calls
                if len(tool_calls) != len(tool_outputs):
                    # If the tool calls changed, mark them as an error
                    tool_outputs = []
                    for i, tool_call in enumerate(tool_calls):
                        tool_outputs.append(
                            {
                                "tool_call_id": tool_call.id,
                                "output": "Error: openai run timed out. You can try again one more time.",
                            }
                        )
                else:
                    # Re-map tool_outputs to the new tool_call IDs
                    for i, tool_name in enumerate(tool_names):
                        for tool_call in tool_calls[:]:
                            if tool_call.function.name == tool_name:
                                tool_outputs[i]["tool_call_id"] = tool_call.id
                                tool_calls.remove(tool_call)
                                break

                self.submit_tool_outputs(tool_outputs, event_handler)
            else:
                raise e

        # Return None so the outer loop continues
        return None

    # -----------------------------
    # Private helper methods
    # -----------------------------

    def _ensure_no_active_run(self, action: str = "wait") -> None:
        """
        Make sure the thread is in a safe state before any mutating call.

        Parameters
        ----------
        action : {"wait", "cancel"}
            "wait"   – block until the current run reaches a terminal state.
            "cancel" – actively cancel the run, then wait until the
                        cancellation is confirmed.
        """
        if not self._run or self._run.status in self.terminal_states:
            return  # already safe

        if action == "cancel":
            self.cancel_run()  # poll() inside guarantees termination
        else:
            self._run_until_done()  # passive wait

        # Defensive sanity-check
        if self._run and self._run.status not in self.terminal_states:
            raise RuntimeError(
                f"Run {self._run.id} still active after _ensure_no_active_run "
                f"(status={self._run.status})."
            )

    def _setup_attachments(
        self,
        attachments: list[Attachment] | None,
        message_files: list[str] | None,
        recipient_agent: Agent,
    ) -> list[Attachment]:
        """Prepare attachments (file_ids, relevant tools) if provided."""
        if not attachments:
            attachments = []

        if message_files:
            recipient_tools = []

            if FileSearch in recipient_agent.tools:
                recipient_tools.append({"type": "file_search"})
            if CodeInterpreter in recipient_agent.tools:
                recipient_tools.append({"type": "code_interpreter"})

            for file_id in message_files:
                attachments.append(
                    Attachment(
                        file_id=file_id,
                        tools=recipient_tools or [{"type": "file_search"}],
                    )
                )

        return attachments

    def _debug_print_sender_and_url(self, recipient_agent):
        """Utility method to print debug info about the conversation.
        Determines the sender's name based on the agent type.
        """
        sender_name = "user" if isinstance(self.agent, User) else self.agent.name
        logger.info(
            f"THREAD:[ {sender_name} -> {recipient_agent.name} ]: URL {self.thread_url}"
        )

    def _try_run_failed_recovery(
        self,
        error_attempts: int,
        recipient_agent: Agent,
        additional_instructions: str | None,
        event_handler: Type[AgencyEventHandler] | None,
        tool_choice: AssistantToolChoice | None,
        response_format: dict | None,
        parent_run_id: str | None,
    ) -> bool:
        """Attempts to recover from run failures if they match common errors (up to 3 times)."""
        error_message = (
            self._run.last_error.message.lower() if self._run.last_error else ""
        )
        common_errors = [
            "something went wrong",
            "the server had an error processing your request",
            "rate limit reached",
        ]
        if error_attempts < 3 and any(e in error_message for e in common_errors):
            if error_attempts < 2:
                time.sleep(1 + error_attempts)
            else:
                # Make one last try with a 'Continue.' user prompt
                self.create_message(message="Continue.", role="user")

            self._create_run(
                recipient_agent,
                additional_instructions,
                event_handler,
                tool_choice,
                response_format=response_format,
            )
            return True
        else:
            # chain error
            self._tracking_manager.track_chain_error(
                error=Exception(f"OpenAI Run Failed. Error: {error_message}"),
                run_id=self._run.id,
                parent_run_id=parent_run_id,
            )
            return False

    def _on_run_incomplete(self, parent_run_id: str | None):
        """Handle incomplete runs by firing chain error callbacks and raising an exception."""
        self._tracking_manager.track_chain_error(
            error=Exception(
                "OpenAI Run Incomplete. Details: " + str(self._run.incomplete_details)
            ),
            run_id=self._run.id,
            parent_run_id=parent_run_id,
        )
        raise Exception(
            "OpenAI Run Incomplete. Details: ", self._run.incomplete_details
        )

    def _validate_assistant_response(
        self,
        recipient_agent: Agent,
        last_message: str,
        validation_attempts: int,
        yield_messages: bool,
        additional_instructions: str | None,
        event_handler: Type[AgencyEventHandler] | None,
        tool_choice: AssistantToolChoice | None,
        response_format: dict | None,
    ):
        """
        If recipient_agent has a response validator, attempt to validate the last_message.
        If validation fails, send the validation prompt back to the agent and re-create run.

        Returns:
        - None if validation passes
        - A dict if validation fails, containing:
            {
            "validation_attempts": int,
            "continue_loop": bool,
            "message_outputs": List[MessageOutput]   (possibly empty if no messages to yield)
            }
        """
        if recipient_agent.response_validator:
            try:
                recipient_agent.response_validator(message=last_message)
            except Exception as e:
                # We only retry if below the maximum number of validation attempts
                if validation_attempts < recipient_agent.validation_attempts:
                    # Attempt to parse the exception as text
                    message_outputs = []
                    try:
                        evaluated_content = eval(str(e))
                        content = (
                            evaluated_content
                            if isinstance(evaluated_content, list)
                            else str(e)
                        )
                    except Exception as e2:
                        content = str(e2)

                    # Create the user message to inform the model about the validation error
                    message_obj = self.create_message(message=content, role="user")

                    # If we were streaming messages, store them to yield in the caller
                    if yield_messages and hasattr(message_obj.content[0], "text"):
                        message_outputs.append(
                            MessageOutput(
                                "text",
                                self.agent.name,
                                recipient_agent.name,
                                message_obj.content[0].text.value,
                                message_obj,
                            )
                        )

                    # Fire the event handler callbacks for the message
                    if event_handler:
                        handler = event_handler()
                        handler.on_message_created(message_obj)
                        handler.on_message_done(message_obj)

                    # Increment validations and recreate run
                    validation_attempts += 1
                    self._create_run(
                        recipient_agent,
                        additional_instructions,
                        event_handler,
                        tool_choice,
                        response_format=response_format,
                    )
                    # Return structured info for the caller
                    return {
                        "validation_attempts": validation_attempts,
                        "continue_loop": True,
                        "message_outputs": message_outputs,
                    }
        return None
