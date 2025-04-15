from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Literal, Union

from openai.lib._tools import pydantic_function_tool
from openai.types.beta.threads.runs.tool_call import ToolCall
from pydantic import BaseModel

from agency_swarm.util.shared_state import SharedState


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


class BaseTool(BaseModel, ABC):
    """Base class for all tools in Agency Swarm.

    Tools can be executed in three ways:
    1. Synchronously via run() - Default execution in the main thread
    2. In parallel via threading - When ToolConfig.async_mode = "threading"
    3. Asynchronously via run_async() - For true async implementations (Future)
    """

    _shared_state: ClassVar[SharedState] = None
    _caller_agent: Any = None
    _event_handler: Any = None
    _tool_call: ToolCall = None
    openai_schema: ClassVar[Dict[str, Any]]

    def __init__(self, **kwargs):
        if not self.__class__._shared_state:
            self.__class__._shared_state = SharedState()
        super().__init__(**kwargs)

        # Ensure all ToolConfig variables are initialized
        config_defaults = {
            "strict": False,
            "one_call_at_a_time": False,
            "output_as_result": False,
            "async_mode": None,
        }

        for key, value in config_defaults.items():
            if not hasattr(self.ToolConfig, key):
                setattr(self.ToolConfig, key, value)

    class ToolConfig:
        strict: bool = False
        one_call_at_a_time: bool = False
        # return the tool output as assistant message
        output_as_result: bool = False
        async_mode: Union[Literal["threading"], None] = None

    @classproperty
    def openai_schema(cls) -> Dict[str, Any]:
        """
        Return the schema using OpenAI's official pydantic_function_tool.

        Note:
            It's important to add a docstring to describe how to best use this class; it will be included in the description attribute and be part of the prompt.

        Returns:
            dict: The OpenAI function schema
        """
        schema = pydantic_function_tool(cls)
        # Flatten the schema - move function properties to top level
        if "function" in schema:
            schema.update(schema.pop("function"))
        return schema

    @abstractmethod
    def run(self) -> Any:
        """Execute the tool synchronously.

        This is the default execution method that must be implemented by all tools.
        For parallel execution, configure ToolConfig.async_mode = "threading".

        Returns:
            The result of the tool execution
        """
        pass

    async def run_async(self) -> Any:
        """Execute the tool asynchronously.

        Override this method to implement true async execution.
        By default falls back to running the sync method in a thread pool.

        Returns:
            The result of the async tool execution
        """
        import asyncio

        return await asyncio.to_thread(self.run)
