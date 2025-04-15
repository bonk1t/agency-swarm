from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import queue
import sys
import threading
import uuid
import warnings
from enum import Enum
from typing import (
    AsyncIterator,
    Literal,
    Type,
    TypeVar,
)

from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.responses.file_search_tool import FileSearchTool
from openai.types.responses.file_search_tool_param import FileSearchToolParam
from openai.types.responses.response_create_params import ToolChoice
from pydantic import BaseModel, Field, field_validator
from rich.console import Console

from agency_swarm.agents import Agent
from agency_swarm.messages import AgentResponse
from agency_swarm.threads import Thread
from agency_swarm.tools import SendMessage, SendMessageBase
from agency_swarm.types import ThreadsCallbacks
from agency_swarm.user import User
from agency_swarm.util.errors import RefusalError
from agency_swarm.util.files import get_file_purpose
from agency_swarm.util.shared_state import SharedState
from agency_swarm.util.streaming import (
    AgencyEventHandler,
    create_gradio_handler,
    create_term_handler,
)
from agency_swarm.util.tracking.tracking_manager import TrackingManager

logger = logging.getLogger(__name__)
console = Console()
T = TypeVar("T", bound=BaseModel)


class Agency:
    """
    A class representing a group of agents that can communicate with each other.
    """

    def __init__(
        self,
        agency_chart: list | None = None,
        entry_points: list[Agent] | None = None,
        communication_flows: list[tuple[Agent, Agent]] | None = None,
        shared_instructions: str = "",
        shared_files: list[str] | None = None,
        async_mode: Literal["threading", "tools_threading"] | None = None,
        send_message_tool_class: Type[SendMessageBase] = SendMessage,
        threads_callbacks: ThreadsCallbacks | None = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        max_prompt_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        truncation_strategy: dict | None = None,
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
            temperature: Default temperature for agents.
            top_p: Default top_p for agents.
            max_prompt_tokens: Default max prompt tokens.
            max_completion_tokens: Default max completion tokens.
            truncation_strategy: Default truncation strategy.
        """
        # Initialize basic attributes
        self.ceo = None
        self.user = User()
        self.agents: list[Agent] = []
        self.entry_points: list[Agent] = []  # Store entry point agents
        self.threads: dict[str, Thread] = {}  # Initialize empty threads dictionary
        self.shared_files = shared_files if shared_files else []
        self.async_mode = async_mode
        self.send_message_tool_class = send_message_tool_class
        self.threads_callbacks = threads_callbacks
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
            self._parse_old_agency_chart(agency_chart)
        else:
            if not entry_points:
                raise ValueError("Must provide at least one entry point agent")
            self._parse_agency_chart(entry_points, communication_flows or [])

        self._init_threads()
        self._create_special_tools()
        self._init_agents_sync()
        self._load_state()

    def _make_thread_key(self, sender: Agent | User, recipient: Agent) -> str:
        """Generate a consistent thread key in the format 'sender->recipient'.

        Args:
            sender: The sender (User or Agent)
            recipient: The recipient Agent

        Returns:
            Thread key string in format 'sender->recipient'
        """
        return f"{sender.name}->{recipient.name}"

    def _parse_thread_key(self, key: str) -> tuple[str, str]:
        """Parse a thread key into sender and recipient names.

        Args:
            key: Thread key in format 'sender->recipient'

        Returns:
            Tuple of (sender_name, recipient_name)

        Raises:
            ValueError: If key format is invalid
        """
        try:
            sender, recipient = key.split("->")
            return sender, recipient
        except ValueError:
            raise ValueError(
                f"Invalid thread key format: {key}. Expected 'sender->recipient'"
            )

    def _init_threads(self):
        """Initialize communication threads between agents.

        Creates threads with consistent 'sender->recipient' key format:
        1. User->Agent threads for entry points (e.g. 'user->AgentName')
        2. Agent->Agent threads from communication flows (e.g. 'AgentA->AgentB')

        Thread keys are always in the format 'sender->recipient' where:
        - sender can be 'user' or an agent name
        - recipient is always an agent name
        """
        # Initialize threads for entry points (user->agent communication)
        for agent in self.entry_points:
            thread_key = self._make_thread_key(User(), agent)
            self.threads[thread_key] = Thread(
                sender=User(), recipient=agent, messages=[]
            )

        # Initialize threads for agent->agent communication
        for initiator, recipient in self.communication_flows:
            thread_key = self._make_thread_key(initiator, recipient)
            self.threads[thread_key] = Thread(
                sender=initiator, recipient=recipient, messages=[]
            )

    def _load_state(self):
        """Load thread states from storage.

        Loads thread messages from persistent storage using the threads_callbacks.
        Thread states are stored with consistent 'sender->recipient' keys.
        """
        if not self.threads_callbacks:
            logger.debug("No threads_callbacks, skipping state loading")
            return

        # Load state directly from callback
        state = self.threads_callbacks["load"]()
        logger.debug(f"Loaded state: {state}")

        if not state:
            logger.debug("No state found")
            return

        thread_state = state.get("threads", {})
        logger.debug(f"Thread state: {thread_state}")

        # Initialize threads with loaded state
        for thread_key, thread in self.threads.items():
            if thread_key in thread_state:
                thread.messages = thread_state[thread_key].get("messages", [])
                logger.debug(
                    f"Loaded {len(thread.messages)} messages into thread {thread_key}"
                )

    def _save_state(self):
        """Save current thread states.

        Saves thread messages to persistent storage using the threads_callbacks.
        Thread states are stored with consistent 'sender->recipient' keys.
        """
        if not self.threads_callbacks:
            return

        # Build state dictionary using consistent thread keys
        thread_state = {}
        for thread_key, thread in self.threads.items():
            thread_state[thread_key] = {
                "messages": thread.messages,
            }

        # Save state through callback
        self.threads_callbacks["save"]({"threads": thread_state})
        logger.debug(f"Saved state for {len(thread_state)} threads")

    def _get_thread(self, recipient_agent: Agent | None = None) -> Thread:
        """Get the appropriate thread for communication with the recipient agent.

        Args:
            recipient_agent: The agent to get the thread for. If None, uses the first entry point.

        Returns:
            Thread: The appropriate Thread instance for communication.

        Raises:
            ValueError: If no thread exists for the given recipient agent or if no entry points defined.
        """
        # If no recipient specified, use first entry point
        if not recipient_agent:
            if not self.entry_points:
                raise ValueError("No entry points defined in the agency.")
            recipient_agent = self.entry_points[0]
            thread_key = self._make_thread_key(User(), recipient_agent)
            return self.threads[thread_key]

        # For user->entry_point communication
        if recipient_agent in self.entry_points:
            thread_key = self._make_thread_key(User(), recipient_agent)
            return self.threads[thread_key]

        # For agent->agent communication
        for initiator, recipient in self.communication_flows:
            if recipient == recipient_agent:
                thread_key = self._make_thread_key(initiator, recipient)
                thread = self.threads.get(thread_key)
                if thread:
                    return thread

        raise ValueError(
            f"No valid communication path exists to {recipient_agent.name}"
        )

    async def get_response(
        self,
        message: str,
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: dict | None = None,
    ) -> str | AsyncIterator[AgentResponse]:
        """Get a response using the Responses API.

        Maintains conversation state and message routing between agents.

        Args:
            message: The message to send (must be a string)
            message_files: Optional list of file IDs
            recipient_agent: Optional specific agent to send to
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice
            response_format: Optional response format

        Returns:
            The agent's response

        Raises:
            ValueError: If message or recipient agent is invalid
        """
        # Validate input
        if not isinstance(message, str):
            raise ValueError("Message must be a string")

        # Get appropriate thread and set up message flow
        thread = self._get_thread(recipient_agent)

        thread.sender = User()

        # Start tracking this response chain if no parent_run_id
        chain_id = self.tracking_manager.start_chain(
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
    ) -> str:
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

        # Get appropriate thread for communication
        thread = self._get_thread(recipient_agent)
        thread.sender = User()

        res = thread.get_response_stream(
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
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
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
        message_files: list[str] | None = None,
        recipient_agent: Agent | None = None,
        additional_instructions: str | None = None,
        attachments: list[dict] | None = None,
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
        recipient_agent_names = [agent.name for agent in self.entry_points]
        recipient_agent = self.entry_points[0]

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
                                thread = self._get_thread(recipient_agent)
                                file = thread.client.files.create(
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
                                if FileSearchTool not in recipient_agent.tools:
                                    recipient_agent.tools.append(FileSearchTool)
                                    print(
                                        "Added FileSearch tool to recipient agent to analyze the file."
                                    )

                                if not recipient_agent.file_search:
                                    recipient_agent.file_search = FileSearchToolParam()
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

                # Check for file attachments and add necessary tools
                self.check_and_add_tools_in_attachments(
                    [a.get("file_id") for a in attachments if a.get("file_id")],
                    recipient_agent,
                )

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
                for agent in self.entry_points
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

        self.recipient_agents = [str(agent.name) for agent in self.entry_points]

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
        """Initialize all agents asynchronously."""
        # Initialize agents from entry points
        for agent in self.entry_points:
            if isinstance(agent, str):
                # If agent is provided as string, instantiate it
                agent_class = self._get_agent_class(agent)
                agent = agent_class(**self.kwargs)
            self.agents.append(agent)

        # Initialize agents from communication flows
        for initiator, recipient in self.communication_flows:
            for agent in (initiator, recipient):
                if agent not in self.agents:
                    if isinstance(agent, str):
                        agent_class = self._get_agent_class(agent)
                        agent = agent_class(**self.kwargs)
                    self.agents.append(agent)

    def _get_agent_class(self, agent_name: str) -> Type[Agent]:
        """
        Resolve an agent class from a string name.

        Args:
            agent_name (str): The name of the agent class to resolve. Can be:
                - A fully qualified import path (e.g. 'my_package.agents.MyAgent')
                - A class name if the agent is in the current module

        Returns:
            Type[Agent]: The resolved agent class

        Raises:
            ImportError: If the agent class cannot be imported
            ValueError: If the agent class is not found or is not a subclass of Agent
        """
        try:
            # First try to import as a fully qualified path
            module_path, class_name = agent_name.rsplit(".", 1)
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            # If that fails, try to find the class in the current module
            try:
                # Get the module where this Agency class is defined
                current_module = sys.modules[self.__class__.__module__]
                agent_class = getattr(current_module, agent_name)
            except AttributeError:
                raise ValueError(f"Could not find agent class '{agent_name}'")

        # Verify the class is a subclass of Agent
        if not issubclass(agent_class, Agent):
            raise ValueError(f"Class '{agent_name}' is not a subclass of Agent")

        return agent_class

    def _init_agents_sync(self):
        """Initialize all agents synchronously."""
        # Initialize agents from entry points
        for agent in self.entry_points:
            if isinstance(agent, str):
                # If agent is provided as string, instantiate it
                agent_class = self._get_agent_class(agent)
                agent = agent_class(**self.kwargs)
            self.agents.append(agent)

        # Initialize agents from communication flows
        for initiator, recipient in self.communication_flows:
            for agent in (initiator, recipient):
                if agent not in self.agents:
                    if isinstance(agent, str):
                        agent_class = self._get_agent_class(agent)
                        agent = agent_class(**self.kwargs)
                    self.agents.append(agent)

    def _parse_old_agency_chart(self, agency_chart):
        """
        Parses the provided agency chart to initialize and organize agents within the agency.
        DEPRECATED: Use entry_points and communication_flows instead.

        Parameters:
            agency_chart: A structure representing the hierarchical organization of agents within the agency.
                    It can contain Agent objects and lists of Agent objects.

        This method iterates through each node in the agency chart. If a node is an Agent, it is set as the CEO if not already assigned.
        If a node is a list, it iterates through the agents in the list, adding them to the agency and establishing communication
        threads between them. It raises an exception if the agency chart is invalid or if multiple CEOs are defined.
        """
        if not isinstance(agency_chart, list):
            raise ValueError("Agency chart must be a list")

        if len(agency_chart) == 0:
            raise ValueError("Agency chart cannot be empty")

        # First pass: Add all agents and identify CEO
        for node in agency_chart:
            if isinstance(node, Agent):
                if not self.ceo:
                    self.ceo = node
                    self._add_agent(self.ceo)
                    self._add_main_recipient(node)
                else:
                    self._add_agent(node)
                    self._add_main_recipient(node)
            elif isinstance(node, list):
                for agent in node:
                    if not isinstance(agent, Agent):
                        raise ValueError(f"Invalid agent in agency chart: {agent}")
                    self._add_agent(agent)
            else:
                raise ValueError(f"Invalid node type in agency chart: {type(node)}")

        # Second pass: Set up communication flows
        for node in agency_chart:
            if isinstance(node, list):
                for i in range(len(node) - 1):
                    current_agent = node[i]
                    next_agent = node[i + 1]

                    # Skip self-communication
                    if current_agent.name == next_agent.name:
                        continue

                    # Add communication flow and thread
                    thread_key = self._make_thread_key(current_agent, next_agent)
                    if thread_key not in self.threads:
                        self.threads[thread_key] = Thread(
                            sender=current_agent, recipient=next_agent, messages=[]
                        )
                        self.communication_flows.append((current_agent, next_agent))

        # Ensure we have a CEO
        if not self.ceo and self.agents:
            self.ceo = self.agents[0]
            self._add_main_recipient(self.ceo)

    def _parse_agency_chart(
        self, entry_points: list[Agent], communication_flows: list[tuple[Agent, Agent]]
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
        for sender, recipient in communication_flows:
            if not isinstance(sender, Agent) or not isinstance(recipient, Agent):
                raise ValueError("Communication flow must be between Agent instances")

            if sender.name == recipient.name:
                raise ValueError(f"Agent {sender.name} cannot communicate with itself")

            # Add agents if not already added
            self._add_agent(sender)
            self._add_agent(recipient)

            # Store communication flow
            self.communication_flows.append((sender, recipient))

            # Set up communication thread
            thread_key = self._make_thread_key(sender, recipient)
            if thread_key not in self.threads:
                self.threads[thread_key] = Thread(
                    sender=sender,
                    recipient=recipient,
                    messages=[],
                )

    def _add_agent(self, agent):
        """
        Adds an agent to the agency, assigning a temporary ID if necessary.

        Parameters:
            agent (Agent): The agent to be added to the agency.

        Returns:
            int: The index of the added agent within the agency's agents list.

        This method adds an agent to the agency's list of agents. If the agent does not have an ID,
        it assigns a temporary unique ID. It checks for uniqueness of the agent's name before addition.
        """
        if not isinstance(agent, Agent):
            raise ValueError(f"Expected Agent instance, got {type(agent)}")

        # Assign temp ID if needed
        if not agent.id:
            agent.id = f"temp_id_{str(uuid.uuid4())}"

        # Check if agent already exists
        for existing_agent in self.agents:
            if existing_agent.id == agent.id:
                return self.agents.index(existing_agent)
            if existing_agent.name == agent.name:
                raise ValueError(f"Agent name '{agent.name}' is already in use")

        # Add new agent
        self.agents.append(agent)
        return len(self.agents) - 1

    def _add_main_recipient(self, agent):
        """
        Adds an agent to the agency's list of main recipients.

        Parameters:
            agent (Agent): The agent to be added to the agency's list of main recipients.

        This method adds an agent to the agency's list of main recipients. These are agents that can be directly contacted by the user.
        """
        main_recipient_ids = [agent.id for agent in self.entry_points]

        if agent.id not in main_recipient_ids:
            self.entry_points.append(agent)

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

        This method iterates through the agents and threads in the agency, creating SendMessage tools for each agent.
        These tools enable agents to send messages to other agents as defined in the agency's structure.
        The SendMessage tools are tailored to the specific recipient agents that each agent can communicate with.

        Thread keys are in the format 'sender->recipient' where:
        - sender can be 'user' or an agent name
        - recipient is always an agent name
        """
        # Iterate through all agents and create their SendMessage tools
        for agent in self.agents:
            # Find all threads where this agent is the sender
            recipient_names = []
            for thread_key in self.threads.keys():
                sender, recipient = self._parse_thread_key(thread_key)
                if sender == agent.name:
                    recipient_names.append(recipient)

            # Get recipient agents and create the SendMessage tool
            recipient_agents = self._get_agents_by_names(recipient_names)
            if len(recipient_agents) > 0:
                agent.add_tool(self._create_send_message_tool(agent, recipient_agents))

    def _create_send_message_tool(self, agent: Agent, recipient_agents: list[Agent]):
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
        SendMessage._threads = self.threads

        return SendMessage

    def _get_agent_by_name(self, name: str) -> Agent:
        """Get an agent by its name.

        Args:
            name: Name of the agent to find

        Returns:
            Agent: The found agent instance

        Special cases:
            - Returns a User instance if name is 'user'
            - Searches through registered agents otherwise

        Raises:
            ValueError: If no agent with the given name is found
        """
        # Special case for user
        if name == "user":
            return User()

        # Search through registered agents
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise ValueError(f"Agent '{name}' not found")

    def _get_agents_by_names(self, agent_names: list[str]) -> list[Agent]:
        """
        Retrieves a list of agent objects based on their names.

        Parameters:
            agent_names: A list of strings representing the names of the agents to be retrieved.

        Returns:
            A list of Agent objects corresponding to the given names.
        """
        return [self._get_agent_by_name(agent_name) for agent_name in agent_names]

    def _get_agent_ids(self):
        """Get list of all agent IDs in the agency."""
        return [agent.id for agent in self.agents]

    def _get_agent_names(self):
        """Get list of all agent names in the agency."""
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

    def _configure_file_search(self, agent: Agent, file_ids: list[str]):
        """Configure FileSearch tool for an agent with given file IDs."""
        if not file_ids:
            return

        if FileSearchTool not in agent.tools:
            print("Adding FileSearch tool for uploaded files...")
            agent.tools.append(FileSearchTool)

        if not agent.file_search:
            agent.file_search = FileSearchToolParam()
        agent.file_search.file_ids = file_ids

    def check_and_add_tools_in_attachments(
        self, file_ids: list[str], recipient_agent: Agent
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
            if not any(isinstance(t, FileSearchTool) for t in recipient_agent.tools):
                recipient_agent.tools.append(FileSearchTool)
                print("Added FileSearch tool to recipient agent to analyze the file.")
