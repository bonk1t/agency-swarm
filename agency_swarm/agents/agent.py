from __future__ import annotations

import inspect
import os
from typing import (
    AsyncIterator,
    Literal,
    Optional,
    Type,
    TypedDict,
    Union,
)

from openai.types.responses.file_search_tool import FileSearchTool
from openai.types.responses.file_search_tool_param import FileSearchToolParam
from openai.types.responses.response_create_params import ToolChoice

from agency_swarm.constants import DEFAULT_MODEL
from agency_swarm.messages import AgentResponse
from agency_swarm.threads import Thread
from agency_swarm.tools import (
    BaseTool,
    SendMessage,
)
from agency_swarm.tools.tool_factory import ToolFactory
from agency_swarm.types import ThreadsCallbacks
from agency_swarm.user import User
from agency_swarm.util.oai import get_openai_client
from agency_swarm.util.openapi import validate_openapi_spec
from agency_swarm.util.streaming.agency_event_handler import AgencyEventHandler


class ExampleMessage(TypedDict):
    """Example message for agent training.

    This class defines the structure of example messages used to train agents.
    Each example message has a role (user or assistant), content, and optional
    metadata.
    """

    role: Literal["user", "assistant"]
    content: str
    metadata: Optional[dict[str, str]]


class Agent:
    """
    A class representing an AI agent that can use tools and communicate with other agents.
    """

    def __init__(
        self,
        name: str = None,
        description: str = "",
        instructions: str = "",
        tools: list[Union[type[BaseTool], type[FileSearchTool]]] = None,
        temperature: float = None,
        top_p: float = 1.0,
        response_format: Union[str, dict, type] = "auto",
        tools_folder: str = None,
        files_folder: Union[list[str], str] = None,
        schemas_folder: Union[list[str], str] = None,
        api_headers: dict[str, dict[str, str]] = None,
        api_params: dict[str, dict[str, str]] = None,
        metadata: dict[str, str] = None,
        model: str = DEFAULT_MODEL,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        validation_attempts: int = 1,
        max_prompt_tokens: int = None,
        max_completion_tokens: int = None,
        truncation_strategy: dict = None,
        examples: list[ExampleMessage] = None,
        file_search: FileSearchToolParam = None,
        parallel_tool_calls: bool = True,
        threads_callbacks: ThreadsCallbacks = None,
    ):
        """Initialize an Agent.

        Args:
            name: Name of the agent.
            description: Description of the agent's purpose.
            instructions: Instructions for the agent.
            tools: List of tools the agent can use.
            temperature: Temperature for responses.
            top_p: Top p sampling parameter.
            response_format: Response format specification.
            tools_folder: Path to tools folder.
            files_folder: Path(s) to files folder.
            schemas_folder: Path(s) to OpenAPI schemas.
            api_headers: Headers for API requests.
            api_params: Parameters for API requests.
            metadata: Additional metadata.
            model: Model to use.
            reasoning_effort: Reasoning effort level.
            validation_attempts: Number of validation attempts.
            max_prompt_tokens: Maximum prompt tokens.
            max_completion_tokens: Maximum completion tokens.
            truncation_strategy: Token truncation strategy.
            examples: Example messages.
            file_search: File search configuration.
            parallel_tool_calls: Whether to allow parallel tool calls.
            threads_callbacks: Optional callbacks for thread persistence.
        """
        # Initialize basic attributes
        self.name = name if name else self.__class__.__name__
        self.role = self.__class__.__name__
        self.description = description
        self.instructions = instructions
        self.tools = tools[:] if tools is not None else []
        self.temperature = temperature
        self.top_p = top_p
        self.response_format = response_format
        self.tools_folder = tools_folder
        self.files_folder = files_folder if files_folder else []
        self.schemas_folder = schemas_folder if schemas_folder else []
        self.api_headers = api_headers if api_headers else {}
        self.api_params = api_params if api_params else {}
        self.metadata = metadata if metadata else {}
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.validation_attempts = validation_attempts
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.truncation_strategy = truncation_strategy
        self.examples = examples
        self.file_search = file_search
        self.parallel_tool_calls = parallel_tool_calls
        self.threads_callbacks = threads_callbacks

        # Initialize internal state
        self._chat_id = None
        self._shared_instructions = None
        self._shared_state = None
        self._file_search_ids = []
        self.id = f"agent_{self.name}_{id(self)}"

        # Initialize OpenAI client
        self.client = get_openai_client()

        # Read instructions if provided
        self._read_instructions()

        # Parse tools and schemas
        self._parse_schemas()
        self._parse_tools_folder()

    @property
    def shared_state(self):
        return self._shared_state

    @shared_state.setter
    def shared_state(self, value):
        self._shared_state = value
        for tool in self.tools:
            if issubclass(tool, BaseTool):
                tool._shared_state = value

    def response_validator(self, message: str | list) -> str:
        """
        Validates the response from the agent. If the response is invalid, it must raise an exception with instructions
        for the caller agent on how to proceed.

        Parameters:
            message (str): The response from the agent.

        Returns:
            str: The validated response.
        """
        return message

    @property
    def chat_id(self) -> str:
        """Get or generate chat ID for this agent's conversations."""
        if self._chat_id is None:
            self._chat_id = f"agent_{self.name}_{id(self)}"
        return self._chat_id

    async def get_response(
        self,
        message: str,
        message_files: list[str] | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        text_only: bool = False,
    ) -> str | AgentResponse:
        """Get a response from this agent using the Responses API.

        Note: SendMessage tools will not work without Agency context.
        For multi-agent communication, use Agency class instead.

        Args:
            message: The message to send
            message_files: Optional list of file IDs
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice
            text_only: Whether to return only the text content

        Returns:
            str if text_only=True, otherwise AgentResponse
        """
        # Check for SendMessage tools
        if any(issubclass(t, SendMessage) for t in self.tools):
            raise ValueError(
                "This agent has SendMessage tools which require Agency context. "
                "Please use Agency class instead."
            )

        # Create thread with state management
        thread = Thread(
            sender=User(),
            recipient=self,
            messages=self.messages,
        )

        response = await thread.get_response(
            message=message,
            message_files=message_files,
            additional_instructions=additional_instructions,
            tool_choice=tool_choice,
        )
        return response.content if text_only else response

    async def get_response_stream(
        self,
        message: str,
        event_handler: Type[AgencyEventHandler],
        message_files: list[str] | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        text_only: bool = False,
    ) -> AsyncIterator[str | AgentResponse]:
        """Get a streaming response from this agent.

        Note: SendMessage tools will not work without Agency context.
        For multi-agent communication, use Agency class instead.

        Args:
            message: The message to send
            event_handler: Handler for streaming events
            message_files: Optional list of file IDs
            additional_instructions: Optional additional instructions
            tool_choice: Optional tool choice
            text_only: Whether to yield only text content

        Yields:
            str if text_only=True, otherwise AgentResponse
        """
        # Check for SendMessage tools
        if any(issubclass(t, SendMessage) for t in self.tools):
            raise ValueError(
                "This agent has SendMessage tools which require Agency context. "
                "Please use Agency class instead."
            )

        # Create thread with state management
        thread = Thread(
            sender=User(),
            recipient=self,
            messages=self.messages,
        )

        async for response in thread.get_response_stream(
            message=message,
            event_handler=event_handler,
            message_files=message_files,
            additional_instructions=additional_instructions,
            tool_choice=tool_choice,
        ):
            yield response.content if text_only else response

    def get_response_sync(
        self,
        message: str,
        message_files: list[str] | None = None,
        additional_instructions: str | None = None,
        tool_choice: ToolChoice | None = None,
        text_only: bool = False,
    ) -> str | AgentResponse:
        """Synchronous version of get_response.

        Note: Streaming is not supported in sync mode.
        Use get_response_stream for streaming responses.
        """
        import asyncio

        return asyncio.run(
            self.get_response(
                message=message,
                message_files=message_files,
                additional_instructions=additional_instructions,
                tool_choice=tool_choice,
                text_only=text_only,
            )
        )

    async def _upload_files(self):
        """Upload files and attach IDs to filenames.

        This method:
        1. Uploads files to OpenAI with purpose="file_search"
        2. Stores file IDs in the filename
        3. Configures FileSearch tool with the uploaded file IDs

        Returns:
            List of uploaded file IDs

        Raises:
            Exception: If file upload fails
        """

        def add_id_to_file(f_path, id):
            """Add file id to file name"""
            if os.path.isfile(f_path):
                file_name, file_ext = os.path.splitext(f_path)
                f_path_new = file_name + "_" + id + file_ext
                os.rename(f_path, f_path_new)
                return f_path_new

        def get_id_from_file(f_path):
            """Get file id from file name"""
            if os.path.isfile(f_path):
                file_name, file_ext = os.path.splitext(f_path)
                file_name = os.path.basename(file_name)
                file_name = file_name.split("_")
                if len(file_name) > 1:
                    return file_name[-1] if "file-" in file_name[-1] else None
                else:
                    return None

        files_folders = (
            self.files_folder
            if isinstance(self.files_folder, list)
            else [self.files_folder]
        )

        file_search_ids = []
        upload_errors = []

        for files_folder in files_folders:
            if not isinstance(files_folder, str):
                print(
                    "Files folder path must be a string or list of strings. Skipping... ",
                    files_folder,
                )
                continue

            f_path = files_folder
            if not os.path.isdir(f_path):
                f_path = os.path.join(self.get_class_folder_path(), files_folder)
                f_path = os.path.normpath(f_path)

            if not os.path.isdir(f_path):
                print(f"Files folder '{f_path}' is not a directory. Skipping...")
                continue

            f_paths = os.listdir(f_path)
            f_paths = [f for f in f_paths if not f.startswith(".")]
            f_paths = [os.path.join(f_path, f) for f in f_paths]

            for f_path in f_paths:
                try:
                    f_path = f_path.strip()
                    file_id = get_id_from_file(f_path)

                    if file_id:
                        print(
                            "File already uploaded. Skipping... "
                            + os.path.basename(f_path)
                        )
                        file_search_ids.append(file_id)
                        continue

                    print("Uploading new file... " + os.path.basename(f_path))

                    try:
                        with open(f_path, "rb") as f:
                            file = await self.client.with_options(
                                timeout=80 * 1000,
                            ).files.create(file=f, purpose="file_search")
                            file_id = file.id
                            f.close()  # fix permission error on windows
                    except Exception as e:
                        raise Exception(f"Failed to upload file {f_path}: {str(e)}")

                    try:
                        add_id_to_file(f_path, file_id)
                    except OSError as e:
                        print(
                            f"Warning: Could not rename file {f_path} with ID: {str(e)}"
                        )
                        # Don't fail the whole upload if rename fails

                    file_search_ids.append(file_id)

                except Exception as e:
                    upload_errors.append((f_path, str(e)))
                    print(f"Error uploading file {f_path}: {str(e)}")

        # Report any upload errors
        if upload_errors:
            error_msg = "The following files failed to upload:\n"
            for f_path, error in upload_errors:
                error_msg += f"- {f_path}: {error}\n"
            print(error_msg)

        # Configure FileSearch with uploaded file IDs if we have any
        if file_search_ids:
            if FileSearchTool not in self.tools:
                print("Adding FileSearch tool for uploaded files...")
                self.add_tool(FileSearchTool)

            # Configure FileSearch with uploaded file IDs
            if not self.file_search:
                self.file_search = FileSearchToolParam()
            self.file_search.file_ids = file_search_ids

        return file_search_ids

    def initialize_files_sync(self):
        """Synchronous wrapper for init_files.

        Returns:
            List of uploaded file IDs

        Raises:
            Exception: If file upload fails
        """
        import asyncio

        return asyncio.run(self.init_files())

    async def init_files(self):
        """Initialize files asynchronously.

        This method should be called after __init__ if file handling is needed.

        Returns:
            List of uploaded file IDs

        Raises:
            Exception: If file upload fails
        """
        return await self._upload_files()

    # --- Tool Methods ---

    # TODO: fix 2 methods below
    def add_tool(self, tool):
        if not isinstance(tool, type):
            raise Exception("Tool must not be initialized.")

        subclasses = [FileSearchTool]
        for subclass in subclasses:
            if issubclass(tool, subclass):
                if not any(issubclass(t, subclass) for t in self.tools):
                    self.tools.append(tool)
                return

        if issubclass(tool, BaseTool):
            if tool.__name__ == "ExampleTool":
                print("Skipping importing ExampleTool...")
                return
            self.tools = [t for t in self.tools if t.__name__ != tool.__name__]
            self.tools.append(tool)
        else:
            raise Exception("Invalid tool type.")

    def get_oai_tools(self):
        tools = []
        for tool in self.tools:
            if not isinstance(tool, type):
                print(tool)
                raise Exception("Tool must not be initialized.")

            if issubclass(tool, FileSearchTool):
                tools.append(
                    tool(file_search=self.file_search).model_dump(exclude_none=True)
                )
            elif issubclass(tool, BaseTool):
                tools.append({"type": "function", "function": tool.openai_schema})
            else:
                raise Exception("Invalid tool type.")
        return tools

    def _parse_schemas(self):
        schemas_folders = (
            self.schemas_folder
            if isinstance(self.schemas_folder, list)
            else [self.schemas_folder]
        )

        for schemas_folder in schemas_folders:
            if isinstance(schemas_folder, str):
                f_path = schemas_folder

                if not os.path.isdir(f_path):
                    f_path = os.path.join(self.get_class_folder_path(), schemas_folder)
                    f_path = os.path.normpath(f_path)

                if os.path.isdir(f_path):
                    f_paths = os.listdir(f_path)

                    f_paths = [f for f in f_paths if not f.startswith(".")]

                    f_paths = [os.path.join(f_path, f) for f in f_paths]

                    for f_path in f_paths:
                        with open(f_path, "r") as f:
                            openapi_spec = f.read()
                            f.close()  # fix permission error on windows
                        try:
                            validate_openapi_spec(openapi_spec)
                        except Exception as e:
                            print("Invalid OpenAPI schema: " + os.path.basename(f_path))
                            raise e
                        try:
                            headers = None
                            params = None
                            if os.path.basename(f_path) in self.api_headers:
                                headers = self.api_headers[os.path.basename(f_path)]
                            if os.path.basename(f_path) in self.api_params:
                                params = self.api_params[os.path.basename(f_path)]
                            tools = ToolFactory.from_openapi_schema(
                                openapi_spec, headers=headers, params=params
                            )
                        except Exception as e:
                            print(
                                "Error parsing OpenAPI schema: "
                                + os.path.basename(f_path)
                            )
                            raise e
                        for tool in tools:
                            self.add_tool(tool)
                else:
                    print(
                        "Schemas folder path is not a directory. Skipping... ", f_path
                    )
            else:
                print(
                    "Schemas folder path must be a string or list of strings. Skipping... ",
                    schemas_folder,
                )

    def _parse_tools_folder(self):
        if not self.tools_folder:
            return

        if not os.path.isdir(self.tools_folder):
            self.tools_folder = os.path.join(
                self.get_class_folder_path(), self.tools_folder
            )
            self.tools_folder = os.path.normpath(self.tools_folder)

        if os.path.isdir(self.tools_folder):
            f_paths = os.listdir(self.tools_folder)
            f_paths = [
                f for f in f_paths if not f.startswith(".") and not f.startswith("__")
            ]
            f_paths = [os.path.join(self.tools_folder, f) for f in f_paths]
            for f_path in f_paths:
                if not f_path.endswith(".py"):
                    continue
                if os.path.isfile(f_path):
                    try:
                        tool = ToolFactory.from_file(f_path)
                        self.add_tool(tool)
                    except Exception as e:
                        print(
                            f"Error parsing tool file {os.path.basename(f_path)}: {e}. Skipping..."
                        )
                else:
                    print("Items in tools folder must be files. Skipping... ", f_path)
        else:
            print(
                "Tools folder path is not a directory. Skipping... ", self.tools_folder
            )

    def get_openapi_schema(self, url):
        """Get openapi schema that contains all tools from the agent as different api paths."""
        return ToolFactory.get_openapi_schema(self.tools, url)

    # --- Helper Methods ---

    def add_file_ids(
        self,
        file_ids: list[str],
        tool_resource: Literal["file_search"],
    ):
        """Add file IDs to a tool resource.

        Args:
            file_ids: List of file IDs to add
            tool_resource: The tool resource to add files to
        """
        if not file_ids:
            return

        if self.tool_resources is None:
            self.tool_resources = {}

        if tool_resource == "file_search":
            if FileSearchTool not in self.tools:
                raise Exception("FileSearch tool not found in tools.")

            if (
                tool_resource not in self.tool_resources
                or self.tool_resources[tool_resource] is None
            ):
                self.tool_resources[tool_resource] = {
                    "vector_stores": [{"file_ids": file_ids}]
                }
            elif not self.tool_resources[tool_resource].get("vector_store_ids"):
                self.tool_resources[tool_resource]["vector_stores"] = [
                    {"file_ids": file_ids}
                ]
            else:
                vector_store_id = self.tool_resources[tool_resource][
                    "vector_store_ids"
                ][0]
                self.client.vector_stores.file_batches.create(
                    vector_store_id=vector_store_id, file_ids=file_ids
                )
        else:
            raise Exception("Invalid tool resource.")

    def get_settings_path(self):
        return self.settings_path

    def _read_instructions(self):
        class_instructions_path = os.path.normpath(
            os.path.join(self.get_class_folder_path(), self.instructions)
        )
        if os.path.isfile(class_instructions_path):
            with open(class_instructions_path, "r") as f:
                self.instructions = f.read()
        elif os.path.isfile(self.instructions):
            with open(self.instructions, "r") as f:
                self.instructions = f.read()
        elif (
            "./instructions.md" in self.instructions
            or "./instructions.txt" in self.instructions
        ):
            raise Exception("Instructions file not found.")

    def get_class_folder_path(self):
        try:
            # First, try to use the __file__ attribute of the module
            return os.path.abspath(os.path.dirname(self.__module__.__file__))
        except (TypeError, OSError, AttributeError) as e:
            # If that fails, fall back to inspect
            try:
                class_file = inspect.getfile(self.__class__)
            except (TypeError, OSError, AttributeError) as e:
                return "./"
            return os.path.abspath(os.path.realpath(os.path.dirname(class_file)))

    def add_shared_instructions(self, instructions: str):
        if not instructions:
            return

        if self._shared_instructions is None:
            self._shared_instructions = instructions
        else:
            self.instructions = self.instructions.replace(self._shared_instructions, "")
            self.instructions = self.instructions.strip().strip("\n")
            self._shared_instructions = instructions

        self.instructions = self._shared_instructions + "\n\n" + self.instructions

    # --- Cleanup Methods ---
    def delete(self):
        """Delete all resources associated with this agent."""
        self._delete_files()

    def _delete_files(self):
        """Delete all files associated with this agent."""
        if not self.tool_resources:
            return

        file_ids = []
        if self.tool_resources.get("file_search"):
            file_search_vector_store_ids = self.tool_resources["file_search"].get(
                "vector_store_ids", []
            )
            for vector_store_id in file_search_vector_store_ids:
                files = self.client.vector_stores.files.list(
                    vector_store_id=vector_store_id, limit=100
                )
                for file in files:
                    file_ids.append(file.id)

                self.client.vector_stores.delete(vector_store_id)

        for file_id in file_ids:
            self.client.files.delete(file_id)
