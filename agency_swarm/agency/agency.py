from __future__ import annotations

import asyncio
import inspect
import json
import os
import queue
import threading
import uuid
import warnings
from enum import Enum
from typing import (
    AsyncIterator,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.responses.response_create_params import ToolChoice
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

from agency_swarm.agents import Agent
from agency_swarm.messages import AgentResponse, MessageOutput
from agency_swarm.threads import Thread, ThreadsCallbacks
from agency_swarm.tools import FileSearch, SendMessage
from agency_swarm.tools.oai.file_search import FileSearchConfig
from agency_swarm.tools.send_message.send_message_base import SendMessageBase
from agency_swarm.tracking import AgencyEventHandler, TrackingManager
from agency_swarm.user import User
from agency_swarm.util.errors import RefusalError
from agency_swarm.util.files import get_file_purpose
from agency_swarm.util.shared_state import SharedState
from agency_swarm.util.streaming import (
    create_gradio_handler,
    create_term_handler,
)

console = Console()
T = TypeVar("T", bound=BaseModel)


class Agency:
    """
    A class representing a group of agents that can communicate with each other.
    """

    def __init__(
        self,
        agency_chart: List = None,  # Kept for backwards compatibility
        entry_points: List[Agent] = None,
        communication_flows: List[Tuple[Agent, Agent]] = None,
        shared_instructions: str = "",
        shared_files: Union[str, List[str]] = None,
        async_mode: Literal["threading", "tools_threading"] = None,
        send_message_tool_class: Type[SendMessageBase] = SendMessage,
        threads_callbacks: ThreadsCallbacks = None,
        chat_id: str | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_prompt_tokens: int = None,
        max_completion_tokens: int = None,
        truncation_strategy: dict = None,
    ):
        """Initialize the Agency object.

        Args:
            agency_chart: DEPRECATED - The old structure defining agent hierarchy. Use entry_points and communication_flows instead.
            entry_points: List of agents that users can communicate with directly.
            communication_flows: List of (initiator, recipient) tuples defining allowed communication paths.
            shared_instructions: Path to shared instructions file.
            shared_files: Path(s) to shared files.
            async_mode: Mode for async processing.
            send_message_tool_class: Class for send_message tool.
            threads_callbacks: Callbacks for thread persistence.
            chat_id: Optional identifier for the conversation context.
            temperature: Default temperature for agents.
            top_p: Default top_p for agents.
            max_prompt_tokens: Default max prompt tokens.
            max_completion_tokens: Default max completion tokens.
            truncation_strategy: Default truncation strategy.
        """
        # Initialize basic attributes
        self.ceo = None
        self.user = User()
        self.agents = []
        self.agents_and_threads = {}
        self.main_recipients = []  # Entry point agents
        self.main_thread = None
        self.recipient_agents = None
        self.shared_files = shared_files if shared_files else []
        self.async_mode = async_mode
        self.send_message_tool_class = send_message_tool_class
        self.threads_callbacks = threads_callbacks
        self.chat_id = chat_id or str(uuid.uuid4())
        self.temperature = temperature
        self.top_p = top_p
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.truncation_strategy = truncation_strategy
        self.communication_flows = []  # Store communication flows

        # Initialize shared state
        self.shared_state = SharedState()
        self.tracking_manager = TrackingManager()

        # Handle async mode configuration
        if self.async_mode == "threading":
            from agency_swarm.tools.send_message import SendMessageAsyncThreading

            warnings.warn(
                "'threading' mode is deprecated. Please use send_message_tool_class = SendMessageAsyncThreading for async communication.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.send_message_tool_class = SendMessageAsyncThreading
        elif self.async_mode == "tools_threading":
            Thread.async_mode = "tools_threading"
            warnings.warn(
                "'tools_threading' mode is deprecated. Use tool.ToolConfig.async_mode = 'threading' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Read shared instructions
        if os.path.isfile(
            os.path.join(self._get_class_folder_path(), shared_instructions)
        ):
            self._read_instructions(
                os.path.join(self._get_class_folder_path(), shared_instructions)
            )
        elif os.path.isfile(shared_instructions):
            self._read_instructions(shared_instructions)
        else:
            self.shared_instructions = shared_instructions

        # Parse agency structure
        if agency_chart:
            warnings.warn(
                "agency_chart is deprecated. Please use entry_points and communication_flows instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._parse_agency_chart(agency_chart)
        else:
            if not entry_points:
                raise ValueError("Must provide at least one entry point agent")
            self._parse_new_structure(entry_points, communication_flows or [])

        self._init_threads()
        self._create_special_tools()
        self._init_agents_sync()

        # Load state if chat_id provided
        if chat_id and self.threads_callbacks:
            self._load_state()

    def _load_state(self):
        """Load thread states from storage."""
        if not self.threads_callbacks:
            return

        state = self.threads_callbacks["load"](self.chat_id)
        if not state:
            return

        thread_state = state.get("agent_pairs", {})

        # Initialize threads with loaded state
        for agent_name, threads in self.agents_and_threads.items():
            if agent_name == "main_thread":
                continue
            for other_agent, thread in threads.items():
                state_key = f"{agent_name}->{other_agent}"
                if state_key in thread_state:
                    thread.previous_response_id = thread_state[state_key].get(
                        "previous_response_id"
                    )
                    thread.messages = thread_state[state_key].get("messages", [])

        # Handle main thread separately
        main_thread_state = thread_state.get("main_thread", {})
        if main_thread_state:
            self.main_thread.previous_response_id = main_thread_state.get(
                "previous_response_id"
            )
            self.main_thread.messages = main_thread_state.get("messages", [])

    def _save_state(self):
        """Save current thread states."""
        if not self.threads_callbacks:
            return

        thread_state = {}

        # Save state for all threads
        for agent_name, threads in self.agents_and_threads.items():
            if agent_name == "main_thread":
                continue
            for other_agent, thread in threads.items():
                state_key = f"{agent_name}->{other_agent}"
                thread_state[state_key] = {
                    "previous_response_id": thread.previous_response_id,
                    "messages": thread.messages,
                }

        # Save main thread state
        thread_state["main_thread"] = {
            "previous_response_id": self.main_thread.previous_response_id,
            "messages": self.main_thread.messages,
        }

        # Save to storage
        self.threads_callbacks["save"]({self.chat_id: {"agent_pairs": thread_state}})

    def _init_threads(self):
        """Initialize all threads according to communication flows."""
        # Load thread state if callbacks exist
        thread_state = {}
        if self.threads_callbacks:
            state = self.threads_callbacks["load"](self.chat_id)
            thread_state = state.get(self.chat_id, {}).get("agent_pairs", {})

        # Initialize main thread for first entry point (CEO)
        main_thread_state = thread_state.get("main_thread", {})
        self.main_thread = Thread(
            agent=self.user,
            recipient_agent=self.ceo,
            previous_response_id=main_thread_state.get("previous_response_id"),
            messages=main_thread_state.get("messages", []),
        )
        self.agents_and_threads["main_thread"] = self.main_thread

        # Initialize user->entry_point threads
        for agent in self.main_recipients:
            if agent.name not in self.agents_and_threads:
                self.agents_and_threads[agent.name] = {}

            # Create thread from user to entry point
            thread_key = f"user->{agent.name}"
            state = thread_state.get(thread_key, {})
            self.agents_and_threads[agent.name]["user"] = Thread(
                agent=self.user,
                recipient_agent=agent,
                previous_response_id=state.get("previous_response_id"),
                messages=state.get("messages", []),
            )

        # Initialize agent->agent threads from communication flows
        for initiator, recipient in self.communication_flows:
            if initiator.name not in self.agents_and_threads:
                self.agents_and_threads[initiator.name] = {}

            thread_key = f"{initiator.name}->{recipient.name}"
            state = thread_state.get(thread_key, {})

            self.agents_and_threads[initiator.name][recipient.name] = Thread(
                agent=initiator,
                recipient_agent=recipient,
                previous_response_id=state.get("previous_response_id"),
                messages=state.get("messages", []),
            )

    def _get_thread(self, recipient_agent: Optional["Agent"] = None) -> Thread:
        """Get the appropriate thread for communication with the recipient agent.

        This method ONLY retrieves existing threads - it does not create them.
        All threads should be initialized in _init_threads().

        Args:
            recipient_agent: The agent to get the thread for. If None, uses the first main recipient.

        Returns:
            Thread: The appropriate Thread instance for communication.

        Raises:
            ValueError: If no thread exists for the given recipient agent.
        """
        if not recipient_agent:
            if not self.main_recipients:
                raise ValueError("No main recipients defined in the agency.")
            recipient_agent = self.main_recipients[0]
            return self.main_thread

        # For user->entry_point communication
        if recipient_agent in self.main_recipients:
            return self.agents_and_threads[recipient_agent.name]["user"]

        # For agent->agent communication
        for agent_name, threads in self.agents_and_threads.items():
            if agent_name == "main_thread":
                continue
            if recipient_agent.name in threads:
                thread = threads[recipient_agent.name]
                if not isinstance(thread, Thread):
                    raise RuntimeError(
                        f"Invalid thread type for {recipient_agent.name}: {type(thread)}"
                    )
                return thread

        raise ValueError(f"Cannot communicate directly with {recipient_agent.name}")

    async def get_response(
        self,
        message: Union[str, List[Dict[str, str]]],
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: dict | None = None,
        parent_run_id: str | None = None,
    ) -> str | AsyncIterator[AgentResponse]:
        """Get a response using the Responses API.

        Maintains conversation state and message routing between agents.

        Args:
            message: The message to send
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice
            response_format: Optional response format
            parent_run_id: Optional parent run ID for tracking

        Returns:
            The agent's response

        Raises:
            ValueError: If message or recipient agent is invalid
        """
        # Validate input
        if not isinstance(message, (str, list)):
            raise ValueError("Message must be a string or list of messages")

        if recipient_agent and not isinstance(recipient_agent, Agent):
            raise ValueError("Invalid recipient agent")

        # Get appropriate thread and set up message flow
        thread = self._get_thread(recipient_agent)

        # If message is sent directly to an agent (not through another agent),
        # we need to use the User as the sender
        if recipient_agent and not isinstance(thread.agent, User):
            thread.agent = User()

        # Start tracking this response chain if no parent_run_id
        chain_id = parent_run_id or self.tracking_manager.start_chain(
            message,
            f"Agency: {recipient_agent.name if recipient_agent else 'main'} chain start",
        )

        try:
            # Get response
            response = await thread.get_response(
                message=message,
                message_files=message_files,
                recipient_agent=recipient_agent,
                additional_instructions=additional_instructions,
                tool_choice=tool_choice,
                parent_run_id=chain_id,
            )

            # Save state after successful response
            self._save_state()

            # Track successful completion
            self.tracking_manager.end_chain(response, chain_id)

            return response

        except Exception as e:
            # Track error
            self.tracking_manager.track_chain_error(e, chain_id)
            raise

    def get_response_sync(
        self,
        message: str,
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: dict | None = None,
        parent_run_id: str | None = None,
    ) -> str | AgentResponse:
        """Synchronous version of get_response."""
        return asyncio.run(
            self.get_response(
                message=message,
                message_files=message_files,
                recipient_agent=recipient_agent,
                additional_instructions=additional_instructions,
                tool_choice=tool_choice,
                response_format=response_format,
                parent_run_id=parent_run_id,
            )
        )

    def get_completion(
        self,
        message: str,
        message_files: list[str] | None = None,
        yield_messages: bool = False,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        attachments: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        verbose: bool = False,
        response_format: dict | None = None,
    ) -> Generator[MessageOutput, None, str] | str:
        """DEPRECATED: Use get_response instead.

        This method is kept for backwards compatibility but will be removed in a future version.
        """
        warnings.warn(
            "get_completion is deprecated and will be removed in a future version. "
            "Use get_response instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if attachments:
            raise NotImplementedError(
                "attachments parameter is no longer supported. Use message_files instead."
            )

        if yield_messages:
            raise NotImplementedError(
                "yield_messages=True is no longer supported. Use get_response or get_response_stream instead."
            )

        return self.get_response(
            message=message,
            message_files=message_files,
            recipient_agent=recipient_agent,
            additional_instructions=additional_instructions,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    def get_response_stream(
        self,
        message: str,
        event_handler: Type[AgencyEventHandler],
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Get a streaming response using the Responses API.

        Args:
            message: The message to send
            event_handler: The event handler class to handle the response stream
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice
            response_format: Optional response format

        Returns:
            The final response text after streaming completes
        """
        if not inspect.isclass(event_handler):
            raise Exception("Event handler must not be an instance.")

        chain_id = self.tracking_manager.start_chain(message, "Agency: chain start")

        res = self.main_thread.get_response_stream(
            message=message,
            event_handler=event_handler,
            message_files=message_files,
            recipient_agent=recipient_agent,
            additional_instructions=additional_instructions,
            tool_choice=tool_choice,
            response_format=response_format,
            parent_run_id=chain_id,
        )

        while True:
            try:
                next(res)
            except StopIteration as e:
                event_handler.on_all_streams_end()
                self.tracking_manager.end_chain(e.value, chain_id)
                return e.value
            except Exception as e:
                self.tracking_manager.track_chain_error(e, chain_id)
                raise e

    def get_response_parse(
        self,
        message: str,
        response_format: Type[T],
        message_files: List[str] = None,
        recipient_agent: Agent = None,
        additional_instructions: str = None,
        tool_choice: ToolChoice | None = None,
    ) -> T:
        """Get a response and parse it using a Pydantic model.

        Args:
            message: The message to send
            response_format: Pydantic model class to parse the response
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice

        Returns:
            The parsed response as an instance of the provided Pydantic model

        Raises:
            RefusalError: If the agent refuses to provide a valid response
            Exception: If the response cannot be parsed into the model
        """
        response_schema = type_to_response_format_param(response_format)

        res = self.get_response(
            message=message,
            message_files=message_files,
            recipient_agent=recipient_agent,
            additional_instructions=additional_instructions,
            tool_choice=tool_choice,
            response_format=response_schema,
        )

        try:
            return response_format.model_validate_json(res)
        except:
            parsed_res = json.loads(res)
            if "refusal" in parsed_res:
                raise RefusalError(parsed_res["refusal"])
            else:
                raise Exception("Failed to parse response: " + res)

    def get_completion_stream(
        self,
        message: str,
        event_handler: Type[AgencyEventHandler],
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        attachments: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: dict | None = None,
    ) -> str:
        """DEPRECATED: Use get_response_stream instead.

        This method is kept for backwards compatibility but will be removed in a future version.
        """
        warnings.warn(
            "get_completion_stream is deprecated and will be removed in a future version. "
            "Use get_response_stream instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.get_response_stream(
            message=message,
            event_handler=event_handler,
            message_files=message_files,
            recipient_agent=recipient_agent,
            additional_instructions=additional_instructions,
            attachments=attachments,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    def get_completion_parse(
        self,
        message: str,
        response_format: Type[T],
        message_files: List[str] = None,
        recipient_agent: Agent = None,
        additional_instructions: str = None,
        attachments: List[dict] = None,
        tool_choice: ToolChoice | None = None,
        verbose: bool = False,
    ) -> T:
        """DEPRECATED: Use get_response_parse instead.

        This method is kept for backwards compatibility but will be removed in a future version.
        """
        warnings.warn(
            "get_completion_parse is deprecated and will be removed in a future version. "
            "Use get_response_parse instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.get_response_parse(
            message=message,
            response_format=response_format,
            message_files=message_files,
            recipient_agent=recipient_agent,
            additional_instructions=additional_instructions,
            attachments=attachments,
            tool_choice=tool_choice,
        )

    def demo_gradio(self, height=450, dark_mode=True, **kwargs):
        """Launch a Gradio-based demo interface for the agency.

        Args:
            height: Height of the chatbot widget. Defaults to 450.
            dark_mode: Whether to use dark mode. Defaults to True.
            **kwargs: Additional arguments passed to gr.Interface.launch()
        """
        try:
            import gradio as gr
        except ImportError:
            raise Exception("Please install gradio: pip install gradio")

        js = """function () {
            gradioURL = window.location.href
            if (!gradioURL.endsWith('?__theme={theme}')) {
                window.location.replace(gradioURL + '?__theme={theme}');
            }
        }"""

        if dark_mode:
            js = js.replace("{theme}", "dark")
        else:
            js = js.replace("{theme}", "light")

        attachments = []
        images = []
        message_file_names = None
        uploading_files = False
        recipient_agent_names = [agent.name for agent in self.main_recipients]
        recipient_agent = self.main_recipients[0]

        chatbot_queue = queue.Queue()
        gradio_handler_class = create_gradio_handler(chatbot_queue=chatbot_queue)

        with gr.Blocks(js=js) as demo:
            chatbot = gr.Chatbot(height=height)
            with gr.Row():
                with gr.Column(scale=9):
                    dropdown = gr.Dropdown(
                        label="Recipient Agent",
                        choices=recipient_agent_names,
                        value=recipient_agent.name,
                    )
                    msg = gr.Textbox(label="Your Message", lines=4)
                with gr.Column(scale=1):
                    file_upload = gr.Files(label="OpenAI Files", type="filepath")
            button = gr.Button(value="Send", variant="primary")

            def handle_dropdown_change(selected_option):
                nonlocal recipient_agent
                recipient_agent = self._get_agent_by_name(selected_option)

            def handle_file_upload(file_list):
                nonlocal attachments
                nonlocal message_file_names
                nonlocal uploading_files
                nonlocal images
                uploading_files = True
                attachments = []
                message_file_names = []
                if file_list:
                    try:
                        for file_obj in file_list:
                            purpose = get_file_purpose(file_obj.name)

                            with open(file_obj.name, "rb") as f:
                                # Upload the file to OpenAI
                                file = self.main_thread.client.files.create(
                                    file=f, purpose=purpose
                                )

                            if purpose == "vision":
                                images.append(
                                    {
                                        "type": "image_file",
                                        "image_file": {"file_id": file.id},
                                    }
                                )
                            else:
                                # Configure FileSearch with the file ID
                                if FileSearch not in recipient_agent.tools:
                                    recipient_agent.tools.append(FileSearch)
                                    print(
                                        "Added FileSearch tool to recipient agent to analyze the file."
                                    )

                                if not recipient_agent.file_search:
                                    recipient_agent.file_search = FileSearchConfig()
                                if not recipient_agent.file_search.file_ids:
                                    recipient_agent.file_search.file_ids = []
                                recipient_agent.file_search.file_ids.append(file.id)

                            message_file_names.append(file.filename)
                            print(f"Uploaded file ID: {file.id}")
                        return attachments
                    except Exception as e:
                        print(f"Error: {e}")
                        return str(e)
                    finally:
                        uploading_files = False

                uploading_files = False
                return "No files uploaded"

            def user(user_message, history):
                if not user_message.strip():
                    return user_message, history

                nonlocal message_file_names
                nonlocal uploading_files
                nonlocal images
                nonlocal attachments
                nonlocal recipient_agent

                # Check if attachments contain file search or code interpreter types
                def check_and_add_tools_in_attachments(attachments, recipient_agent):
                    for attachment in attachments:
                        for tool in attachment.get("tools", []):
                            if tool["type"] == "file_search":
                                if not any(
                                    isinstance(t, FileSearch)
                                    for t in recipient_agent.tools
                                ):
                                    recipient_agent.tools.append(FileSearch)
                                    recipient_agent.client.beta.assistants.update(
                                        recipient_agent.id,
                                        tools=recipient_agent.get_oai_tools(),
                                    )
                                    print(
                                        "Added FileSearch tool to recipient agent to analyze the file."
                                    )
                    return None

                check_and_add_tools_in_attachments(attachments, recipient_agent)

                if history is None:
                    history = []

                original_user_message = user_message

                # Append the user message with a placeholder for bot response
                if recipient_agent:
                    user_message = (
                        f"👤 User 🗣️ @{recipient_agent.name}:\n" + user_message.strip()
                    )
                else:
                    user_message = "👤 User:" + user_message.strip()

                if message_file_names:
                    user_message += "\n\n📎 Files:\n" + "\n".join(message_file_names)

                return original_user_message, history + [[user_message, None]]

            def bot(original_message, history, dropdown):
                nonlocal attachments
                nonlocal message_file_names
                nonlocal recipient_agent
                nonlocal recipient_agent_names
                nonlocal images
                nonlocal uploading_files

                if not original_message:
                    return (
                        "",
                        history,
                        gr.update(
                            value=recipient_agent.name,
                            choices=set([*recipient_agent_names, recipient_agent.name]),
                        ),
                    )

                if uploading_files:
                    history.append([None, "Uploading files... Please wait."])
                    yield (
                        "",
                        history,
                        gr.update(
                            value=recipient_agent.name,
                            choices=set([*recipient_agent_names, recipient_agent.name]),
                        ),
                    )
                    return (
                        "",
                        history,
                        gr.update(
                            value=recipient_agent.name,
                            choices=set([*recipient_agent_names, recipient_agent.name]),
                        ),
                    )

                print("Message files: ", attachments)
                print("Images: ", images)

                if images and len(images) > 0:
                    original_message = [
                        {
                            "type": "text",
                            "text": original_message,
                        },
                        *images,
                    ]

                completion_thread = threading.Thread(
                    target=self.get_response_stream,
                    args=(
                        original_message,
                        gradio_handler_class,
                        [],
                        recipient_agent,
                        "",
                        attachments,
                        None,
                    ),
                )
                completion_thread.start()

                attachments = []
                message_file_names = []
                images = []
                uploading_files = False

                new_message = True
                while True:
                    try:
                        bot_message = chatbot_queue.get(block=True)

                        if bot_message == "[end]":
                            completion_thread.join()
                            break

                        if bot_message == "[new_message]":
                            new_message = True
                            continue

                        if bot_message == "[change_recipient_agent]":
                            new_agent_name = chatbot_queue.get(block=True)
                            recipient_agent = self._get_agent_by_name(new_agent_name)
                            yield (
                                "",
                                history,
                                gr.update(
                                    value=new_agent_name,
                                    choices=set(
                                        [*recipient_agent_names, recipient_agent.name]
                                    ),
                                ),
                            )
                            continue

                        if new_message:
                            history.append([None, bot_message])
                            new_message = False
                        else:
                            history[-1][1] += bot_message

                        yield (
                            "",
                            history,
                            gr.update(
                                value=recipient_agent.name,
                                choices=set(
                                    [*recipient_agent_names, recipient_agent.name]
                                ),
                            ),
                        )
                    except queue.Empty:
                        break

            button.click(user, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
                bot, [msg, chatbot, dropdown], [msg, chatbot, dropdown]
            )
            dropdown.change(handle_dropdown_change, dropdown)
            file_upload.change(handle_file_upload, file_upload)
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                bot, [msg, chatbot, dropdown], [msg, chatbot, dropdown]
            )

            # Enable queuing for streaming intermediate outputs
            demo.queue(default_concurrency_limit=10)

        # Launch the demo
        demo.launch(**kwargs)
        return demo

    def _setup_autocomplete(self):
        """
        Sets up readline with the completer function.
        """
        try:
            import readline
        except ImportError:
            # Attempt to import pyreadline for Windows compatibility
            try:
                import pyreadline as readline
            except ImportError:
                print(
                    "Module 'readline' not found. Autocomplete will not work. If you are using Windows, try installing 'pyreadline3'."
                )
                return

        if not readline:
            return

        def recipient_agent_completer(text, state):
            """
            Autocomplete completer for recipient agent names.
            """
            options = [
                agent
                for agent in self.recipient_agents
                if agent.lower().startswith(text.lower())
            ]
            if state < len(options):
                return options[state]
            else:
                return None

        try:
            readline.set_completer(recipient_agent_completer)
            readline.parse_and_bind("tab: complete")
        except Exception as e:
            print(
                f"Error setting up autocomplete for agents in terminal: {e}. Autocomplete will not work."
            )

    def run_demo(self):
        """
        Executes agency in the terminal with autocomplete for recipient agent names.
        """
        term_handler_class = create_term_handler(agency=self)

        self.recipient_agents = [str(agent.name) for agent in self.main_recipients]

        self._setup_autocomplete()  # Prepare readline for autocomplete

        while True:
            console.rule()
            text = input("👤 USER: ")

            if not text:
                continue

            if text.lower() == "exit":
                break

            recipient_agent = None
            if "@" in text:
                recipient_agent = text.split("@")[1].split(" ")[0]
                text = text.replace(f"@{recipient_agent}", "").strip()
                try:
                    recipient_agent = [
                        agent
                        for agent in self.recipient_agents
                        if agent.lower() == recipient_agent.lower()
                    ][0]
                    recipient_agent = self._get_agent_by_name(recipient_agent)
                except Exception as e:
                    print(f"Recipient agent {recipient_agent} not found.")
                    continue

            self.get_completion_stream(
                message=text,
                event_handler=term_handler_class,
                recipient_agent=recipient_agent,
            )

    def get_customgpt_schema(self, url: str):
        """Returns the OpenAPI schema for the agency from the CEO agent, that you can use to integrate with custom gpts.

        Parameters:
            url (str): Your server url where the api will be hosted.
        """

        return self.ceo.get_openapi_schema(url)

    def plot_agency_chart(self):
        pass

    async def _init_agents(self):
        """Initialize all agents in the agency."""
        for agent in self.agents:
            if "temp_id" in agent.id:
                agent.id = None

            agent.add_shared_instructions(self.shared_instructions)

            if self.shared_files:
                if isinstance(self.shared_files, str):
                    self.shared_files = [self.shared_files]

                if isinstance(agent.files_folder, str):
                    agent.files_folder = [agent.files_folder]
                    agent.files_folder += self.shared_files
                elif isinstance(agent.files_folder, list):
                    agent.files_folder += self.shared_files

            # Initialize agent configuration
            if self.temperature is not None and agent.temperature is None:
                agent.temperature = self.temperature
            if self.top_p and agent.top_p is None:
                agent.top_p = self.top_p
            if self.max_prompt_tokens is not None and agent.max_prompt_tokens is None:
                agent.max_prompt_tokens = self.max_prompt_tokens
            if (
                self.max_completion_tokens is not None
                and agent.max_completion_tokens is None
            ):
                agent.max_completion_tokens = self.max_completion_tokens
            if (
                self.truncation_strategy is not None
                and agent.truncation_strategy is None
            ):
                agent.truncation_strategy = self.truncation_strategy

            if not agent.shared_state:
                agent.shared_state = self.shared_state

            # Initialize files
            await agent.init_files()

    def _init_agents_sync(self):
        """Synchronous wrapper for _init_agents"""
        return asyncio.run(self._init_agents())

    def _parse_agency_chart(self, agency_chart):
        """
        Parses the provided agency chart to initialize and organize agents within the agency.

        Parameters:
            agency_chart: A structure representing the hierarchical organization of agents within the agency.
                    It can contain Agent objects and lists of Agent objects.

        This method iterates through each node in the agency chart. If a node is an Agent, it is set as the CEO if not already assigned.
        If a node is a list, it iterates through the agents in the list, adding them to the agency and establishing communication
        threads between them. It raises an exception if the agency chart is invalid or if multiple CEOs are defined.
        """
        if not isinstance(agency_chart, list):
            raise Exception("Invalid agency chart.")

        if len(agency_chart) == 0:
            raise Exception("Agency chart cannot be empty.")

        for node in agency_chart:
            if isinstance(node, Agent):
                if not self.ceo:
                    self.ceo = node
                    self._add_agent(self.ceo)
                else:
                    self._add_agent(node)
                self._add_main_recipient(node)

            elif isinstance(node, list):
                for i, agent in enumerate(node):
                    if not isinstance(agent, Agent):
                        raise Exception("Invalid agency chart.")

                    index = self._add_agent(agent)

                    if i == len(node) - 1:
                        continue

                    if agent.name not in self.agents_and_threads.keys():
                        self.agents_and_threads[agent.name] = {}

                    if i < len(node) - 1:
                        other_agent = node[i + 1]
                        if other_agent.name == agent.name:
                            continue
                        if (
                            other_agent.name
                            not in self.agents_and_threads[agent.name].keys()
                        ):
                            self.agents_and_threads[agent.name][other_agent.name] = {
                                "agent": agent.name,
                                "recipient_agent": other_agent.name,
                            }
            else:
                raise Exception("Invalid agency chart.")

    def _parse_new_structure(
        self, entry_points: List[Agent], communication_flows: List[Tuple[Agent, Agent]]
    ):
        """Parse the new agency structure with entry points and communication flows."""
        # Validate entry points
        if not entry_points:
            raise ValueError("Must provide at least one entry point agent")

        # Add entry points
        for agent in entry_points:
            if not isinstance(agent, Agent):
                raise ValueError(f"Entry point {agent} must be an Agent instance")
            self._add_agent(agent)
            self._add_main_recipient(agent)

        # First entry point becomes CEO for backwards compatibility
        self.ceo = entry_points[0]

        # Store and validate communication flows
        self.communication_flows = []
        for initiator, recipient in communication_flows:
            if not isinstance(initiator, Agent) or not isinstance(recipient, Agent):
                raise ValueError("Communication flow must be between Agent instances")

            if initiator.name == recipient.name:
                raise ValueError(
                    f"Agent {initiator.name} cannot communicate with itself"
                )

            # Add agents if not already added
            self._add_agent(initiator)
            self._add_agent(recipient)

            # Store communication flow
            self.communication_flows.append((initiator, recipient))

            # Set up communication thread structure
            if initiator.name not in self.agents_and_threads:
                self.agents_and_threads[initiator.name] = {}

            if recipient.name not in self.agents_and_threads[initiator.name]:
                self.agents_and_threads[initiator.name][recipient.name] = {
                    "agent": initiator.name,
                    "recipient_agent": recipient.name,
                }

    def _add_agent(self, agent):
        """
        Adds an agent to the agency, assigning a temporary ID if necessary.

        Parameters:
            agent (Agent): The agent to be added to the agency.

        Returns:
            int: The index of the added agent within the agency's agents list.

        This method adds an agent to the agency's list of agents. If the agent does not have an ID, it assigns a temporary unique ID. It checks for uniqueness of the agent's name before addition. The method returns the index of the agent in the agency's agents list, which is used for referencing the agent within the agency.
        """
        if not agent.id:
            # assign temp id
            agent.id = "temp_id_" + str(uuid.uuid4())
        if agent.id not in self._get_agent_ids():
            if agent.name in self._get_agent_names():
                raise Exception("Agent names must be unique.")
            self.agents.append(agent)
            return len(self.agents) - 1
        else:
            return self._get_agent_ids().index(agent.id)

    def _add_main_recipient(self, agent):
        """
        Adds an agent to the agency's list of main recipients.

        Parameters:
            agent (Agent): The agent to be added to the agency's list of main recipients.

        This method adds an agent to the agency's list of main recipients. These are agents that can be directly contacted by the user.
        """
        main_recipient_ids = [agent.id for agent in self.main_recipients]

        if agent.id not in main_recipient_ids:
            self.main_recipients.append(agent)

    def _read_instructions(self, path):
        """
        Reads shared instructions from a specified file and stores them in the agency.

        Parameters:
            path (str): The file path from which to read the shared instructions.

        This method opens the file located at the given path, reads its contents, and stores these contents in the 'shared_instructions' attribute of the agency. This is used to provide common guidelines or instructions to all agents within the agency.
        """
        path = path
        with open(path, "r") as f:
            self.shared_instructions = f.read()

    def _create_special_tools(self):
        """
        Creates and assigns 'SendMessage' tools to each agent based on the agency's structure.

        This method iterates through the agents and threads in the agency, creating SendMessage tools for each agent. These tools enable agents to send messages to other agents as defined in the agency's structure. The SendMessage tools are tailored to the specific recipient agents that each agent can communicate with.

        No input parameters.

        No output parameters; this method modifies the agents' toolset internally.
        """
        for agent_name, threads in self.agents_and_threads.items():
            if agent_name == "main_thread":
                continue
            recipient_names = list(threads.keys())
            recipient_agents = self._get_agents_by_names(recipient_names)
            if len(recipient_agents) == 0:
                continue
            agent = self._get_agent_by_name(agent_name)
            agent.add_tool(self._create_send_message_tool(agent, recipient_agents))

    def _create_send_message_tool(self, agent: Agent, recipient_agents: List[Agent]):
        """
        Creates a SendMessage tool to enable an agent to send messages to specified recipient agents.


        Parameters:
            agent (Agent): The agent who will be sending messages.
            recipient_agents (List[Agent]): A list of recipient agents who can receive messages.

        Returns:
            SendMessage: A SendMessage tool class that is dynamically created and configured for the given agent and its recipient agents. This tool allows the agent to send messages to the specified recipients, facilitating inter-agent communication within the agency.
        """
        recipient_names = [agent.name for agent in recipient_agents]
        recipients = Enum("recipient", {name: name for name in recipient_names})

        agent_descriptions = ""
        for recipient_agent in recipient_agents:
            if not recipient_agent.description:
                continue
            agent_descriptions += recipient_agent.name + ": "
            agent_descriptions += recipient_agent.description + "\n"

        class SendMessage(self.send_message_tool_class):
            recipient: recipients = Field(..., description=agent_descriptions)

            @field_validator("recipient")
            @classmethod
            def check_recipient(cls, value):
                if value.value not in recipient_names:
                    raise ValueError(
                        f"Recipient {value} is not valid. Valid recipients are: {recipient_names}"
                    )
                return value

        SendMessage._caller_agent = agent
        SendMessage._agents_and_threads = self.agents_and_threads

        return SendMessage

    def _get_agent_by_name(self, agent_name):
        """
        Retrieves an agent from the agency based on the agent's name.

        Parameters:
            agent_name (str): The name of the agent to be retrieved.

        Returns:
            Agent: The agent object with the specified name.

        Raises:
            Exception: If no agent with the given name is found in the agency.
        """
        # Special case for user
        if agent_name == "user":
            return self.user

        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        raise Exception(f"Agent {agent_name} not found.")

    def _get_agents_by_names(self, agent_names):
        """
        Retrieves a list of agent objects based on their names.

        Parameters:
            agent_names: A list of strings representing the names of the agents to be retrieved.

        Returns:
            A list of Agent objects corresponding to the given names.
        """
        return [self._get_agent_by_name(agent_name) for agent_name in agent_names]

    def _get_agent_ids(self):
        """
        Retrieves the IDs of all agents currently in the agency.

        Returns:
            List[str]: A list containing the unique IDs of all agents.
        """
        return [agent.id for agent in self.agents]

    def _get_agent_names(self):
        """
        Retrieves the names of all agents in the agency.

        Returns:
            List[str]: A list of names of all agents currently part of the agency.
        """
        return [agent.name for agent in self.agents]

    def _get_class_folder_path(self):
        """
        Retrieves the absolute path of the directory containing the class file.

        Returns:
            str: The absolute path of the directory where the class file is located.
        """
        return os.path.abspath(os.path.dirname(inspect.getfile(self.__class__)))

    def delete(self):
        """
        This method deletes the agency and all its agents, cleaning up any files and vector stores associated with each agent.
        """
        for agent in self.agents:
            agent.delete()

    @property
    def _thread_type(self):
        """Get the thread type based on async mode."""
        if self.async_mode == "threading":
            return Thread
        return Thread

    def _configure_file_search(self, agent: Agent, file_ids: List[str]):
        """Configure FileSearch tool for an agent with given file IDs."""
        if not file_ids:
            return

        if FileSearch not in agent.tools:
            print("Adding FileSearch tool for uploaded files...")
            agent.tools.append(FileSearch)

        if not agent.file_search:
            agent.file_search = FileSearchConfig()
        agent.file_search.file_ids = file_ids

    def check_and_add_tools_in_attachments(
        self, file_ids: List[str], recipient_agent: Agent
    ) -> None:
        """Check and add tools based on file types.

        Args:
            file_ids: List of file IDs to check
            recipient_agent: Agent to add tools to
        """
        if not file_ids:
            return

        # Add FileSearch tool if needed
        if any(file_id.startswith("file-") for file_id in file_ids):
            if not any(isinstance(t, FileSearch) for t in recipient_agent.tools):
                recipient_agent.tools.append(FileSearch)
                print("Added FileSearch tool to recipient agent to analyze the file.")
