"""
Test the additional_instructions parameter functionality for Agent and Agency.

This module tests the core additional_instructions feature to ensure:
1. Instructions are temporarily modified during execution
2. Original instructions are properly restored after execution
3. The parameter is correctly passed through Agency to Agent
4. Both get_response and get_response_stream handle additional_instructions
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agents import RunResult

from agency_swarm import Agency, Agent


@pytest.fixture
def base_agent():
    """Create a test agent with known instructions."""
    return Agent(name="TestAgent", instructions="Original agent instructions")


@pytest.fixture
def base_agency(base_agent):
    """Create a test agency with the test agent."""
    return Agency(base_agent)


@pytest.fixture
def mock_run_result():
    """Create a mock RunResult for mocking Runner.run."""
    mock_result = MagicMock(spec=RunResult)
    mock_result.final_output = "Test response"
    mock_result.new_items = []
    return mock_result


@pytest.mark.asyncio
async def test_agent_get_response_modifies_instructions_temporarily(base_agent, mock_run_result):
    """Test that Agent.get_response temporarily modifies instructions with additional_instructions."""
    original_instructions = base_agent.instructions
    additional_text = "Additional test instructions"

    # Track instruction changes during execution
    instruction_history = []

    async def mock_runner_run(*args, **kwargs):
        instruction_history.append(base_agent.instructions)
        return mock_run_result

    with patch("agents.Runner.run", side_effect=mock_runner_run):
        await base_agent.get_response(message="Test message", additional_instructions=additional_text)

    # Verify instructions were modified during execution
    assert len(instruction_history) == 1
    modified_instructions = instruction_history[0]
    assert additional_text in modified_instructions
    assert original_instructions in modified_instructions

    # Verify original instructions are restored
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agent_get_response_restores_instructions_on_error(base_agent):
    """Test that original instructions are restored even if Runner.run fails."""
    original_instructions = base_agent.instructions
    additional_text = "Additional test instructions"

    from agents import AgentsException

    with patch("agents.Runner.run", side_effect=RuntimeError("Test error")):
        with pytest.raises(AgentsException):  # Now using specific exception type
            await base_agent.get_response(message="Test message", additional_instructions=additional_text)

    # Verify original instructions are restored despite the error
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agent_get_response_stream_modifies_instructions_temporarily(base_agent):
    """Test that Agent.get_response_stream temporarily modifies instructions with additional_instructions."""
    original_instructions = base_agent.instructions
    additional_text = "Additional streaming instructions"

    # Track instruction changes during execution
    instruction_history = []

    async def mock_stream_events():
        instruction_history.append(base_agent.instructions)
        yield {"event": "text", "data": "test"}

    mock_streamed_result = MagicMock()
    mock_streamed_result.stream_events = mock_stream_events

    with patch("agents.Runner.run_streamed", return_value=mock_streamed_result):
        events = []
        async for event in base_agent.get_response_stream(
            message="Test message", additional_instructions=additional_text
        ):
            events.append(event)

    # Verify instructions were modified during execution
    assert len(instruction_history) == 1
    modified_instructions = instruction_history[0]
    assert additional_text in modified_instructions
    assert original_instructions in modified_instructions

    # Verify original instructions are restored
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agent_get_response_stream_restores_instructions_on_error(base_agent):
    """Test that original instructions are restored even if streaming fails."""
    original_instructions = base_agent.instructions
    additional_text = "Additional streaming instructions"

    with patch("agents.Runner.run_streamed", side_effect=RuntimeError("Streaming error")):
        try:
            async for _event in base_agent.get_response_stream(
                message="Test message", additional_instructions=additional_text
            ):
                pass
        except Exception:
            pass  # Expected to fail

    # Verify original instructions are restored despite the error
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agency_get_response_passes_additional_instructions(base_agency, base_agent):
    """Test that Agency.get_response passes additional_instructions to the target agent."""
    additional_text = "Agency additional instructions"

    with patch.object(base_agent, "get_response", new_callable=AsyncMock) as mock_get_response:
        mock_get_response.return_value = MagicMock(final_output="Agency response")

        await base_agency.get_response(message="Test message", additional_instructions=additional_text)

        # Verify additional_instructions was passed to agent
        mock_get_response.assert_called_once()
        call_kwargs = mock_get_response.call_args[1]
        assert call_kwargs["additional_instructions"] == additional_text


@pytest.mark.asyncio
async def test_agency_get_response_stream_passes_additional_instructions(base_agency, base_agent):
    """Test that Agency.get_response_stream passes additional_instructions to the target agent."""
    additional_text = "Agency streaming instructions"

    async def mock_stream():
        yield {"event": "text", "data": "test"}

    with patch.object(base_agent, "get_response_stream", return_value=mock_stream()) as mock_stream_method:
        events = []
        async for event in base_agency.get_response_stream(
            message="Test message", additional_instructions=additional_text
        ):
            events.append(event)

        # Verify additional_instructions was passed to agent
        mock_stream_method.assert_called_once()
        call_kwargs = mock_stream_method.call_args[1]
        assert call_kwargs["additional_instructions"] == additional_text


@pytest.mark.asyncio
async def test_agent_get_response_without_additional_instructions(base_agent, mock_run_result):
    """Test that Agent.get_response works normally without additional_instructions."""
    original_instructions = base_agent.instructions

    with patch("agents.Runner.run", return_value=mock_run_result):
        await base_agent.get_response(message="Test message")

    # Verify instructions were not modified
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agent_get_response_empty_additional_instructions(base_agent, mock_run_result):
    """Test that empty additional_instructions doesn't modify instructions."""
    original_instructions = base_agent.instructions

    with patch("agents.Runner.run", return_value=mock_run_result):
        await base_agent.get_response(message="Test message", additional_instructions="")

    # Verify instructions were not modified for empty string
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agent_get_response_none_additional_instructions(base_agent, mock_run_result):
    """Test that None additional_instructions doesn't modify instructions."""
    original_instructions = base_agent.instructions

    with patch("agents.Runner.run", return_value=mock_run_result):
        await base_agent.get_response(message="Test message", additional_instructions=None)

    # Verify instructions were not modified for None
    assert base_agent.instructions == original_instructions


@pytest.mark.asyncio
async def test_agent_with_none_original_instructions(mock_run_result):
    """Test additional_instructions works when agent has no original instructions."""
    agent = Agent(name="NoInstructionsAgent", instructions=None)
    additional_text = "Only additional instructions"

    # Track instruction changes during execution
    instruction_history = []

    async def mock_runner_run(*args, **kwargs):
        instruction_history.append(agent.instructions)
        return mock_run_result

    with patch("agents.Runner.run", side_effect=mock_runner_run):
        await agent.get_response(message="Test message", additional_instructions=additional_text)

    # Verify instructions were set to additional_instructions during execution
    assert len(instruction_history) == 1
    assert instruction_history[0] == additional_text

    # Verify original None instructions are restored
    assert agent.instructions is None


def test_deprecated_get_completion_passes_additional_instructions(base_agency, base_agent):
    """Test that the deprecated get_completion method passes additional_instructions."""
    additional_text = "Deprecated method instructions"

    with patch.object(base_agency, "_async_get_completion", new_callable=AsyncMock) as mock_async_get_completion:
        mock_async_get_completion.return_value = "Completion response"

        # Suppress the deprecation warning for this test
        with pytest.warns(DeprecationWarning):
            result = base_agency.get_completion(message="Test message", additional_instructions=additional_text)

        # Verify the result
        assert result == "Completion response"

        # Verify additional_instructions was passed to _async_get_completion
        mock_async_get_completion.assert_called_once()
        call_kwargs = mock_async_get_completion.call_args[1]
        assert call_kwargs["additional_instructions"] == additional_text
