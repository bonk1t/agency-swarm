import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import RunContextWrapper, RunResult
from agents.items import ToolCallItem
from agents.stream_events import RunItemStreamEvent
from pydantic import Field

from agency_swarm import Agent, BaseTool
from agency_swarm.context import MasterContext
from agency_swarm.thread import ThreadManager
from agency_swarm.tools.send_message import SendMessage

# --- Fixtures ---


@pytest.fixture
def mock_sender_agent():
    agent = MagicMock(spec=Agent)
    agent.name = "SenderAgent"
    return agent


@pytest.fixture
def mock_recipient_agent(mock_run_context_wrapper):
    agent = MagicMock(spec=Agent)
    agent.name = "RecipientAgent"
    # Provide minimal required args for RunResult
    mock_run_result = RunResult(
        _last_agent=agent,
        input=[],
        new_items=[],
        raw_responses=[],
        input_guardrail_results=[],
        output_guardrail_results=[],
        final_output="Response from recipient",
        context_wrapper=mock_run_context_wrapper,
    )
    agent.get_response = AsyncMock(return_value=mock_run_result)
    return agent


@pytest.fixture
def mock_master_context():
    context = MagicMock(spec=MasterContext)
    context.user_context = {"user_key": "user_value"}
    return context


@pytest.fixture
def mock_run_context_wrapper(mock_master_context):
    wrapper = MagicMock(spec=RunContextWrapper)
    wrapper.context = mock_master_context
    return wrapper


@pytest.fixture
def mock_context(mock_sender_agent, mock_recipient_agent):
    context = MagicMock(spec=MasterContext)
    context.agents = {"SenderAgent": mock_sender_agent, "RecipientAgent": mock_recipient_agent}
    context.thread_manager = MagicMock(spec=ThreadManager)
    context.thread_manager.get_thread = MagicMock(return_value=MagicMock())
    context.thread_manager.add_items_and_save = AsyncMock()
    context.user_context = {"user_key": "user_val"}
    context.shared_instructions = None
    return context


@pytest.fixture
def mock_wrapper(mock_context, mock_sender_agent):
    wrapper = MagicMock(spec=RunContextWrapper)
    wrapper.context = mock_context
    wrapper.hooks = MagicMock()
    wrapper.agent = mock_sender_agent
    return wrapper


@pytest.fixture
def specific_send_message_tool(mock_sender_agent, mock_recipient_agent):
    # Create an instance of SendMessage for testing its on_invoke_tool method directly
    return SendMessage(
        sender_agent=mock_sender_agent,
        recipients={mock_recipient_agent.name.lower(): mock_recipient_agent},
    )


@pytest.fixture
def legacy_tool():
    # Create a class that inherits from BaseTool
    class TestTool(BaseTool):
        input: str = Field(description="The input to the tool")

        class ToolConfig:
            strict = True

        def run(self):
            print(f"Running TestTool with input: {self.input}")
            return self.input

    return TestTool


# --- Test Cases ---


@pytest.mark.asyncio
async def test_send_message_success(specific_send_message_tool, mock_wrapper, mock_recipient_agent, mock_context):
    message_content = "Test message"
    args_dict = {
        "recipient_agent": mock_recipient_agent.name,  # Add the recipient_agent field
        "my_primary_instructions": "Primary instructions for test.",
        "message": message_content,
        "additional_instructions": "Additional instructions for test.",
    }
    args_json_string = json.dumps(args_dict)

    result = await specific_send_message_tool.on_invoke_tool(
        wrapper=mock_wrapper, arguments_json_string=args_json_string
    )

    assert result == "Response from recipient"
    # Check that get_response was called with the expected parameters
    mock_recipient_agent.get_response.assert_called_once()
    call_args = mock_recipient_agent.get_response.call_args
    assert call_args.kwargs["message"] == message_content
    assert call_args.kwargs["sender_name"] == specific_send_message_tool.sender_agent.name
    assert "additional_instructions" in call_args.kwargs
    assert "agency_context" in call_args.kwargs


@pytest.mark.asyncio
async def test_send_message_invalid_json(specific_send_message_tool, mock_wrapper):
    args_json_string = "{invalid json string"
    expected_error_message = (
        f"Error: Invalid arguments format for tool {specific_send_message_tool.name}. Expected a valid JSON string."
    )

    with patch("agency_swarm.tools.send_message.logger") as mock_module_logger:
        result = await specific_send_message_tool.on_invoke_tool(
            wrapper=mock_wrapper, arguments_json_string=args_json_string
        )

    assert result == expected_error_message
    mock_module_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_send_message_missing_required_param(specific_send_message_tool, mock_wrapper):
    # Test missing 'message'
    args_dict_missing_message = {
        "recipient_agent": "RecipientAgent",
        "my_primary_instructions": "Primary instructions.",
        # "message" is missing
    }
    args_json_missing_message = json.dumps(args_dict_missing_message)
    expected_error_missing_message = (
        f"Error: Missing required parameter 'message' for tool {specific_send_message_tool.name}."
    )

    with patch("agency_swarm.tools.send_message.logger") as mock_module_logger:
        result = await specific_send_message_tool.on_invoke_tool(
            wrapper=mock_wrapper, arguments_json_string=args_json_missing_message
        )
    assert result == expected_error_missing_message
    mock_module_logger.error.assert_called_once_with(
        f"Tool '{specific_send_message_tool.name}' invoked without 'message' parameter."
    )

    mock_module_logger.reset_mock()

    # Test missing 'my_primary_instructions'
    args_dict_missing_instr = {
        "recipient_agent": "RecipientAgent",
        "message": "A message",
        # my_primary_instructions is missing
    }
    args_json_missing_instr = json.dumps(args_dict_missing_instr)
    expected_error_missing_instr = (
        f"Error: Missing required parameter 'my_primary_instructions' for tool {specific_send_message_tool.name}."
    )

    with patch("agency_swarm.tools.send_message.logger") as mock_module_logger_instr:
        result = await specific_send_message_tool.on_invoke_tool(
            wrapper=mock_wrapper, arguments_json_string=args_json_missing_instr
        )
    assert result == expected_error_missing_instr
    mock_module_logger_instr.error.assert_called_once_with(
        f"Tool '{specific_send_message_tool.name}' invoked without 'my_primary_instructions' parameter."
    )


@pytest.mark.asyncio
async def test_send_message_target_agent_error(specific_send_message_tool, mock_wrapper, mock_recipient_agent):
    error_text = "Target agent failed"
    mock_recipient_agent.get_response.side_effect = RuntimeError(error_text)
    message_content = "Test message"
    args_dict = {
        "recipient_agent": mock_recipient_agent.name,
        "my_primary_instructions": "Primary instructions.",
        "message": message_content,
        "additional_instructions": "",
    }
    args_json_string = json.dumps(args_dict)
    expected_error_message = (
        f"Error: Failed to get response from agent '{mock_recipient_agent.name}'. Reason: {error_text}"
    )

    with patch("agency_swarm.tools.send_message.logger") as mock_module_logger:
        result = await specific_send_message_tool.on_invoke_tool(
            wrapper=mock_wrapper, arguments_json_string=args_json_string
        )

    assert result == expected_error_message
    mock_module_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_emit_send_message_start_emits_event_with_type(specific_send_message_tool):
    """SendMessage._emit_send_message_start should emit a run_item_stream_event sentinel."""
    streaming_context = MagicMock()
    streaming_context.put_event = AsyncMock()

    await specific_send_message_tool._emit_send_message_start(
        streaming_context=streaming_context,
        sender_agent_name="SenderAgent",
        thread_manager=None,
        arguments_json="{}",
    )

    event = streaming_context.put_event.call_args[0][0]
    assert isinstance(event, RunItemStreamEvent)
    assert event.type == "run_item_stream_event"
    assert event.name == "tool_called"
    assert isinstance(event.item, ToolCallItem)
    assert event.item.type == "tool_call_item"


@pytest.mark.asyncio
async def test_legacy_tool(legacy_tool):
    """
    Test that BaseTool can be used via the on_invoke_tool method of the adapted FunctionTool.
    """
    from agency_swarm.tools.ToolFactory import ToolFactory

    function_tool = ToolFactory.adapt_base_tool(legacy_tool)
    input_json = '{"input": "hello"}'
    result = await function_tool.on_invoke_tool(None, input_json)
    assert result == "hello"


@pytest.mark.asyncio
async def test_basetool_context_support():
    """Test that BaseTools receive context through _context attribute."""
    from pydantic import Field

    from agency_swarm import BaseTool
    from agency_swarm.tools.ToolFactory import ToolFactory

    # Create a BaseTool that uses context
    class ContextAwareTool(BaseTool):
        """A tool that accesses context."""

        message: str = Field(..., description="Message to process")

        def run(self):
            if self.context is not None:
                # Access user context using the cleaner API
                user_val = self.context.get("test_key", "no_value")
                return f"Message: {self.message}, Context: {user_val}"
            else:
                return f"Message: {self.message}, Context: None"

    # Adapt to FunctionTool
    function_tool = ToolFactory.adapt_base_tool(ContextAwareTool)

    # Create mock context
    mock_master_context = MagicMock()
    mock_master_context.user_context = {"test_key": "test_value"}
    mock_master_context.get = lambda key, default=None: mock_master_context.user_context.get(key, default)

    mock_wrapper = MagicMock()
    mock_wrapper.context = mock_master_context

    # Test with context
    input_json = '{"message": "Hello"}'
    result = await function_tool.on_invoke_tool(mock_wrapper, input_json)
    assert result == "Message: Hello, Context: test_value"

    # Test without context (None)
    result_no_ctx = await function_tool.on_invoke_tool(None, input_json)
    assert result_no_ctx == "Message: Hello, Context: None"


@pytest.mark.asyncio
async def test_schema_conversion():
    agent = Agent(name="test", instructions="test", schemas_folder="tests/data/schemas")
    tool_names = [tool.name for tool in agent.tools]
    assert "getTimeByTimezone" in tool_names


def test_tools_folder_autoload():
    tools_path = Path("tests/data/tools").resolve()
    agent = Agent(name="test", instructions="test", tools_folder=str(tools_path))
    tool_names = [tool.name for tool in agent.tools]
    assert "ExampleTool1" in tool_names
    assert "sample_tool" in tool_names


def test_relative_tools_folder_is_class_local():
    agent = Agent(name="test", instructions="test", tools_folder="../data/tools")
    tool_names = [tool.name for tool in agent.tools]
    assert "ExampleTool1" in tool_names and "sample_tool" in tool_names


def test_tools_folder_edge_cases(tmp_path):
    """Test tools_folder handles edge cases correctly."""
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()

    # Create files that should be ignored
    (tools_dir / "_private_tool.py").write_text("# Should be ignored")
    (tools_dir / "readme.txt").write_text("Not a Python file")
    (tools_dir / "invalid_tool.py").write_text("invalid python syntax !")

    # Create valid tool
    (tools_dir / "valid_tool.py").write_text("""
from agents import function_tool

@function_tool
def valid_tool() -> str:
    return "works"
""")

    agent = Agent(name="test", instructions="test", tools_folder=str(tools_dir))
    tool_names = [tool.name for tool in agent.tools]

    # Only valid_tool should be loaded
    assert "valid_tool" in tool_names
    assert "_private_tool" not in tool_names
    assert len(tool_names) == 1


def test_tools_folder_none():
    """Test agent works with no tools_folder."""
    agent = Agent(name="test", instructions="test", tools_folder=None)
    assert agent.tools == []


def test_tools_folder_nonexistent_path():
    """Test agent handles nonexistent tools_folder gracefully."""
    agent = Agent(name="test", instructions="test", tools_folder="/nonexistent/path")
    assert agent.tools == []


@pytest.mark.asyncio
async def test_shared_state_property(mock_run_context_wrapper):
    class TestTool(BaseTool):
        def run(self):
            return "ok"

    tool = TestTool()
    tool._context = mock_run_context_wrapper
    with pytest.deprecated_call():
        assert tool._shared_state is mock_run_context_wrapper.context


# TODO: Add tests for response validation aspects
# TODO: Add tests for context/hooks propagation (more complex, might need integration tests)
# TODO: Add parameterized tests for various message inputs (empty, long, special chars)
# TODO: Add tests for specific schema validation failures (if FunctionTool provides hooks)
