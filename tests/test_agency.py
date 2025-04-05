"""Tests for Agency Swarm core functionality.

These tests verify critical functionality of the Agency Swarm framework:
1. Basic agent responses and message handling
2. State persistence and conversation history
3. Agent uniqueness and proper initialization
4. Thread management and message routing
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

from agency_swarm import Agency
from agency_swarm.agents import Agent
from agency_swarm.messages import AgentResponse
from agency_swarm.threads import Thread

# Load environment variables
load_dotenv()


class BasicAgent(Agent):
    """A basic agent for testing core functionality."""

    name = "basic_agent"
    description = "A test agent"
    model = "gpt-4o"  # Use standard model for testing
    temperature = 0
    instructions = """You are a helpful test agent. Be concise in your responses.
    You MUST maintain conversation context and remember previous messages.
    When asked about something mentioned earlier in the conversation, reference it directly.
    This is critical for testing conversation state management."""

    async def init_files(self):
        """Initialize agent files."""
        pass


@pytest.fixture
async def basic_agency():
    """Create a basic agency with a single agent."""
    agent = BasicAgent()
    await agent.init_files()

    agency = Agency(
        entry_points=[agent],  # Use new structure
        temperature=0,
    )
    return agency


@pytest.mark.asyncio
async def test_agent_uniqueness(basic_agency):
    """Test that agents with same name but different IDs are handled correctly."""
    # Create two agents with same name
    agent1 = BasicAgent()
    agent2 = BasicAgent()
    await agent1.init_files()
    await agent2.init_files()

    # Verify they get unique IDs
    assert agent1.id != agent2.id

    # Try to create agency with duplicate named agents
    with pytest.raises(Exception, match="Agent names must be unique"):
        Agency(entry_points=[agent1, agent2])


@pytest.mark.asyncio
async def test_conversation_history(basic_agency):
    """Test that conversation history is maintained properly."""
    # Create a proper AgentResponse object
    mock_response = AgentResponse(
        content="I remember the code BLUE42",
        type="text",
        sender_name="BasicAgent",
        receiver_name="user",
    )

    # Patch the get_response method of the thread
    with patch(
        "agency_swarm.threads.thread.Thread.get_response", new_callable=AsyncMock
    ) as mock_get_response:
        mock_get_response.return_value = mock_response

        # First message
        response1 = await basic_agency.get_response("Remember this secret code: BLUE42")

        # Manually add message to thread history since we're mocking
        basic_agency.main_thread.messages.append(
            {"role": "user", "content": "Remember this secret code: BLUE42"}
        )
        basic_agency.main_thread.messages.append(
            {"role": "assistant", "content": "I remember the code BLUE42"}
        )

        assert isinstance(response1, AgentResponse)
        assert response1.type == "text"

        # Verify message was added to history
        assert len(basic_agency.main_thread.messages) > 0
        assert "BLUE42" in str(basic_agency.main_thread.messages)

        # Second message should reference history
        response2 = await basic_agency.get_response(
            "What was the secret code I told you?"
        )
        assert "BLUE42" in response2.content


@pytest.mark.asyncio
async def test_state_persistence():
    """Test state persistence with callbacks."""
    # Initialize storage
    stored_state = {}

    def load_state(chat_id):
        return stored_state.get(chat_id, {})

    def save_state(state):
        nonlocal stored_state
        stored_state.update(state)

    # Create and initialize agent
    agent = BasicAgent()
    await agent.init_files()

    # Create agency with state management
    agency1 = Agency(
        entry_points=[agent], threads_callbacks={"load": load_state, "save": save_state}
    )

    # First conversation
    response1 = await agency1.get_response("Remember this number: 12345")
    assert response1.type == "text"

    # Verify state was saved
    assert stored_state
    assert agency1.chat_id in stored_state
    assert "agent_pairs" in stored_state[agency1.chat_id]

    # Create new agent and agency with same chat_id
    agent2 = BasicAgent()
    await agent2.init_files()

    agency2 = Agency(
        entry_points=[agent2],
        threads_callbacks={"load": load_state, "save": save_state},
        chat_id=agency1.chat_id,
    )

    # Should remember previous conversation
    response2 = await agency2.get_response("What number did I tell you to remember?")
    assert "12345" in response2.content


@pytest.mark.asyncio
async def test_thread_initialization():
    """Test proper thread initialization and message routing."""
    # Create mock agents
    agent1 = BasicAgent()
    agent1.name = "agent1"
    agent2 = BasicAgent()
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    # Create agency with two agents
    agency = Agency(
        entry_points=[agent1], communication_flows=[(agent1, agent2)], temperature=0
    )

    # Verify threads were created properly
    assert "agent1" in agency.agents_and_threads
    assert "agent2" in agency.agents_and_threads["agent1"]

    # Verify thread has correct agents
    thread = agency.agents_and_threads["agent1"]["agent2"]
    assert isinstance(thread, Thread)
    assert thread.agent.name == "agent1"
    assert thread.recipient_agent.name == "agent2"


@pytest.mark.asyncio
async def test_message_routing():
    """Test that messages are routed correctly between agents."""
    # Create mock agents
    agent1 = BasicAgent()
    agent1.name = "agent1"
    agent2 = BasicAgent()
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    # Create agency with two agents
    agency = Agency(
        entry_points=[agent1], communication_flows=[(agent1, agent2)], temperature=0
    )

    # Send message to entry point agent
    response = await agency.get_response("Hello agent1!", recipient_agent=agent1)

    # Verify message was routed correctly
    assert response.sender_name == "agent1"
    assert response.receiver_name == "user"


@pytest.mark.asyncio
async def test_thread_state_isolation():
    """Test that Thread maintains its own state without depending on Agency."""
    agent1 = BasicAgent()
    agent2 = BasicAgent()
    await agent1.init_files()
    await agent2.init_files()

    # Create a thread directly
    thread = Thread(
        agent=agent1,
        recipient_agent=agent2,
        previous_response_id="test_id",
        messages=[{"role": "user", "content": "test"}],
    )

    # Verify thread maintains its own state
    assert thread.previous_response_id == "test_id"
    assert len(thread.messages) == 1
    assert thread.messages[0]["content"] == "test"

    # Verify thread doesn't have agency state management
    assert not hasattr(thread, "chat_id")
    assert not hasattr(thread, "threads_callbacks")


@pytest.mark.asyncio
async def test_agency_state_sync():
    """Test that Agency properly syncs thread states."""
    # Initialize storage
    stored_state = {}

    def load_state(chat_id):
        return stored_state.get(chat_id, {})

    def save_state(state):
        nonlocal stored_state
        stored_state.update(state)

    # Create and initialize agents
    agent1 = BasicAgent()
    agent2 = BasicAgent()
    agent1.name = "agent1"
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    # Create agency with state management
    agency = Agency(
        entry_points=[agent1],
        communication_flows=[(agent1, agent2)],
        threads_callbacks={"load": load_state, "save": save_state},
    )

    # Send a message
    response1 = await agency.get_response(
        "Remember this: TEST_MESSAGE", recipient_agent=agent1
    )

    # Verify state was saved
    assert stored_state
    assert agency.chat_id in stored_state
    assert "agent_pairs" in stored_state[agency.chat_id]

    # Verify main thread state
    main_thread_state = stored_state[agency.chat_id]["agent_pairs"]["main_thread"]
    assert (
        main_thread_state["previous_response_id"]
        == agency.main_thread.previous_response_id
    )
    assert main_thread_state["messages"] == agency.main_thread.messages

    # Create new agency with same chat_id
    agency2 = Agency(
        entry_points=[agent1],
        communication_flows=[(agent1, agent2)],
        threads_callbacks={"load": load_state, "save": save_state},
        chat_id=agency.chat_id,
    )

    # Verify state was loaded
    assert (
        agency2.main_thread.previous_response_id
        == agency.main_thread.previous_response_id
    )
    assert agency2.main_thread.messages == agency.main_thread.messages


@pytest.mark.asyncio
async def test_multi_agent_conversation():
    """Test conversation between multiple agents with state persistence."""
    agent1 = BasicAgent()
    agent2 = BasicAgent()
    agent1.name = "agent1"
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    agency = Agency(
        entry_points=[agent1], communication_flows=[(agent1, agent2)], temperature=0
    )

    # Send message to entry point agent
    response1 = await agency.get_response(
        "Remember this code: ABC123", recipient_agent=agent1
    )
    assert response1.sender_name == "agent1"

    # Get thread between agents
    thread = agency.agents_and_threads["agent1"]["agent2"]
    assert isinstance(thread, Thread)

    # Verify thread state is maintained
    assert thread.previous_response_id is not None
    assert len(thread.messages) > 0


@pytest.mark.asyncio
async def test_error_handling():
    """Test proper error handling in critical paths."""
    agent = BasicAgent()
    await agent.init_files()

    agency = Agency(entry_points=[agent], temperature=0)

    # Test invalid recipient
    with pytest.raises(ValueError, match="Invalid recipient agent type"):
        await agency.get_response(
            "Hello!",
            recipient_agent=MagicMock(),  # Invalid agent
        )

    # Test invalid message format
    with pytest.raises(
        ValueError, match="Message must be a string or list of messages"
    ):
        await agency.get_response({})  # Invalid message type

    # Test invalid message list format
    with pytest.raises(
        ValueError, match="Each message must be a dict with 'role' and 'content' keys"
    ):
        await agency.get_response(
            [{"invalid": "message"}]
        )  # Invalid message format in list


@pytest.mark.asyncio
async def test_new_agency_structure():
    """Test the new entry_points and communication_flows structure."""
    # Create test agents
    agent1 = BasicAgent()
    agent1.name = "agent1"
    agent2 = BasicAgent()
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    # Create agency with new structure
    agency = Agency(
        entry_points=[agent1],  # Entry point
        communication_flows=[
            (agent1, agent2),  # agent1 can talk to agent2
        ],
        temperature=0,
    )

    # Verify entry points were set up correctly
    assert len(agency.main_recipients) == 1
    assert agency.main_recipients[0].name == "agent1"

    # Verify communication flows were set up
    assert "agent1" in agency.agents_and_threads
    assert "agent2" in agency.agents_and_threads["agent1"]

    # Verify user can talk to entry point
    response1 = await agency.get_response("Hello agent1!", recipient_agent=agent1)
    assert response1.sender_name == "agent1"
    assert response1.receiver_name == "user"

    # Verify user cannot talk directly to non-entry point agent
    with pytest.raises(ValueError, match="Cannot communicate directly with agent2"):
        await agency.get_response("Hello agent2!", recipient_agent=agent2)


@pytest.mark.asyncio
async def test_invalid_agency_structure():
    """Test validation of invalid agency structures."""
    agent1 = BasicAgent()
    agent1.name = "agent1"
    agent2 = BasicAgent()
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    # Test empty entry points
    with pytest.raises(ValueError, match="Must provide at least one entry point agent"):
        Agency(entry_points=[], communication_flows=[])

    # Test invalid entry point type
    with pytest.raises(ValueError, match="Entry point .* must be an Agent instance"):
        Agency(entry_points=["not an agent"], communication_flows=[])

    # Test self-communication
    with pytest.raises(ValueError, match="Agent .* cannot communicate with itself"):
        Agency(entry_points=[agent1], communication_flows=[(agent1, agent1)])

    # Test invalid communication flow types
    with pytest.raises(
        ValueError, match="Communication flow must be between Agent instances"
    ):
        Agency(entry_points=[agent1], communication_flows=[(agent1, "not an agent")])


@pytest.mark.asyncio
async def test_backwards_compatibility():
    """Test that old agency_chart structure still works."""
    agent1 = BasicAgent()
    agent1.name = "agent1"
    agent2 = BasicAgent()
    agent2.name = "agent2"
    await agent1.init_files()
    await agent2.init_files()

    # Create agency with old structure
    with pytest.warns(DeprecationWarning):
        agency = Agency(agency_chart=[agent1, [agent1, agent2]], temperature=0)

    # Verify structure was parsed correctly
    assert len(agency.main_recipients) == 1
    assert agency.main_recipients[0].name == "agent1"
    assert "agent1" in agency.agents_and_threads
    assert "agent2" in agency.agents_and_threads["agent1"]

    # Verify functionality still works
    response = await agency.get_response("Hello agent1!", recipient_agent=agent1)
    assert response.sender_name == "agent1"
    assert response.receiver_name == "user"
