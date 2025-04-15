"""Tests for Agency Swarm core functionality.

These tests verify critical functionality of the Agency Swarm framework:
1. Basic agent responses and message handling
2. State persistence through complete conversation history
3. Agent uniqueness and proper initialization
4. Thread management and message routing
"""

import asyncio
import json
import logging
import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from agency_swarm import Agency
from agency_swarm.agents import Agent
from agency_swarm.messages import AgentResponse
from agency_swarm.threads import Thread

# Load environment variables
load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


class BasicAgent(Agent):
    """A basic agent for testing core functionality."""

    name = "basic_agent"
    description = "A test agent"
    model = "gpt-4o"  # Use standard model for testing
    temperature = 0
    instructions = "You are a helpful test agent. Be concise in your responses."


@pytest_asyncio.fixture
async def basic_agency():
    """Create a basic agency with a single agent."""
    agent = BasicAgent(name="basic_agent")

    agency = Agency(
        entry_points=[agent],
        temperature=0,
    )
    return agency


@pytest.mark.asyncio
async def test_agent_uniqueness(basic_agency):
    """Test that agents with same name but different IDs are handled correctly."""
    # Create two agents with same name
    agent1 = BasicAgent(name="basic_agent")
    agent2 = BasicAgent(name="basic_agent")

    # Verify they get unique IDs
    assert agent1.id != agent2.id

    # Try to create agency with duplicate named agents
    with pytest.raises(
        ValueError, match=f"Agent name '{agent1.name}' is already in use"
    ):
        Agency(entry_points=[agent1, agent2])


@pytest.mark.asyncio
async def test_conversation_history(basic_agency):
    """Test that conversation history is maintained properly."""
    # First message with a specific code to remember
    response1 = await basic_agency.get_response("Remember this secret code: BLUE42")

    assert isinstance(response1, AgentResponse)
    assert response1.type == "text"

    # Verify message was added to history
    assert len(basic_agency.main_thread.messages) > 0

    # Now ask the agent to recall the code
    response2 = await basic_agency.get_response("What was the secret code I told you?")

    # The agent should recall the code from conversation history
    assert "BLUE42" in response2.content.upper()


@pytest.mark.asyncio
async def test_state_persistence():
    """Test that conversation state is maintained through Agency's message history."""
    # Create and initialize agent
    agent = BasicAgent(name="basic_agent")

    # Create single agency instance
    agency = Agency(entry_points=[agent])

    # First conversation
    response1 = await agency.get_response("Remember this number: 12345")
    assert response1.type == "text"

    # Verify message was added to history
    assert len(agency.main_thread.messages) > 0
    assert any(
        "12345" in content_item["text"]
        for msg in agency.main_thread.messages
        for content_item in msg["content"]
    )

    # Send follow-up message to same agency instance
    response2 = await agency.get_response("What number did I ask you to remember?")

    # Verify that the agent remembers the number through agency's message history
    assert "12345" in response2.content

    # Verify message history grew
    assert (
        len(agency.main_thread.messages) > 2
    )  # Should have at least 4 messages now (2 pairs of user/assistant)


@pytest.mark.asyncio
async def test_thread_initialization():
    """Test proper thread initialization and message routing."""
    # Create agents
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2")

    # Create agency with two agents
    agency = Agency(
        entry_points=[agent1], communication_flows=[(agent1, agent2)], temperature=0
    )

    # Verify threads were created properly
    thread_key = f"agent1->agent2"
    assert thread_key in agency.threads

    # Verify thread has correct agents
    thread = agency.threads[thread_key]
    assert isinstance(thread, Thread)
    assert thread.sender.name == "agent1"
    assert thread.recipient.name == "agent2"


@pytest.mark.asyncio
async def test_message_routing():
    """Test that messages are routed correctly between agents and users."""
    # Create agents
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2", instructions="Your name is Secret Agent 007")

    # Create agency with two agents where agent1 can initiate communication with agent2
    agency = Agency(
        entry_points=[agent1], communication_flows=[(agent1, agent2)], temperature=0
    )

    # Test user-to-agent communication
    response = await agency.get_response(
        "Hi agent1, what is your name?", recipient_agent=agent1
    )

    # Verify user-to-agent message routing
    assert response.sender_name == "user"
    assert response.receiver_name == "agent1"

    # Test agent-to-agent communication
    # First, have agent1 initiate communication with agent2
    response = await agency.get_response(
        "Please ask agent2 what their name is and tell me what they said.",
        recipient_agent=agent1,
    )

    # Verify the thread between agents was created
    thread_key = f"agent1->agent2"
    assert thread_key in agency.threads

    # Get the thread between agent1 and agent2
    thread = agency.threads[thread_key]

    # Have agent1 send a direct message to agent2
    agent_response = await thread.get_response("What is your name?")

    # Verify agent-to-agent message routing
    assert agent_response.sender_name == "agent1"
    assert agent_response.receiver_name == "agent2"
    assert "007" in agent_response.content  # Verify agent2 responds with their name

    # Verify the response back to the user has correct routing
    assert response.sender_name == "user"
    assert response.receiver_name == "agent1"
    assert "007" in response.content  # Verify the name was passed back through agent1


@pytest.mark.asyncio
async def test_thread_state_isolation():
    """Test that Thread maintains its own state through message history."""
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2")

    # Create a thread with initial messages
    initial_messages = [
        {"role": "user", "content": [{"type": "input_text", "text": "test message"}]},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "test response"}],
        },
    ]

    thread = Thread(sender=agent1, recipient=agent2, messages=initial_messages)

    # Verify thread maintains its own state
    assert len(thread.messages) == 2
    assert thread.messages[0]["content"][0]["text"] == "test message"
    assert thread.messages[1]["content"][0]["text"] == "test response"


@pytest.mark.asyncio
async def test_agency_state_sync():
    """Test that Agency properly maintains thread states through message history."""
    # Create and initialize agents
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2")

    # Create single agency instance with communication flows
    agency = Agency(
        entry_points=[agent1],
        communication_flows=[(agent1, agent2)],
    )

    # Send a message
    test_message = "Remember this: TEST_MESSAGE"
    response1 = await agency.get_response(test_message, recipient_agent=agent1)

    # Verify main thread state includes messages
    assert any(
        "TEST_MESSAGE" in content_item["text"]
        for msg in agency.main_thread.messages
        for content_item in msg["content"]
    )
    assert any(
        response1.content in content_item["text"]
        for msg in agency.main_thread.messages
        for content_item in msg["content"]
    )

    # Test state persistence by asking about previous message
    response2 = await agency.get_response(
        "What message did I ask you to remember?", recipient_agent=agent1
    )
    assert "TEST_MESSAGE" in response2.content

    # Verify message history grew appropriately
    assert (
        len(agency.main_thread.messages) >= 4
    )  # Should have at least 4 messages (2 pairs)


@pytest.mark.asyncio
async def test_conversation_history_persistence():
    """Test that conversation history is maintained and used for context."""
    agent = BasicAgent(name="basic_agent")

    agency = Agency(entry_points=[agent])

    # Send sequence of related messages
    responses = []

    # First message establishes context
    responses.append(await agency.get_response("My name is Alice."))

    # Second message references first
    responses.append(await agency.get_response("What's my name?"))

    # Verify agent remembers context from history
    assert "Alice" in responses[1].content

    # Verify messages are properly stored
    assert (
        len(agency.main_thread.messages) >= 4
    )  # 2 user messages + 2 assistant responses
    assert any("Alice" in msg["content"] for msg in agency.main_thread.messages)


@pytest.mark.asyncio
async def test_multi_agent_conversation():
    """Test conversation between multiple agents with message history persistence."""
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2")

    # Create single agency instance
    agency = Agency(
        entry_points=[agent1], communication_flows=[(agent1, agent2)], temperature=0
    )

    # Send message to entry point agent
    response1 = await agency.get_response(
        "Remember this code: ABC123", recipient_agent=agent1
    )
    assert response1.sender_name == "agent1"

    # Get thread between agents
    thread_key = f"agent1->agent2"
    thread = agency.threads[thread_key]
    assert isinstance(thread, Thread)

    # Verify thread state is maintained
    assert len(thread.messages) > 0
    assert any("ABC123" in str(msg["content"]) for msg in agency.main_thread.messages)

    # Test state persistence by asking about the code
    response2 = await agency.get_response(
        "What code did I tell you to remember?", recipient_agent=agent1
    )
    assert "ABC123" in response2.content

    # Verify message history grew
    assert len(agency.main_thread.messages) >= 4  # Should have at least 4 messages


@pytest.mark.asyncio
async def test_error_handling():
    """Test proper error handling in critical paths."""
    agent = BasicAgent(name="basic_agent")

    agency = Agency(entry_points=[agent], temperature=0)

    # Test invalid recipient
    class InvalidAgent:
        pass

    with pytest.raises(ValueError, match="Invalid recipient agent type"):
        await agency.get_response(
            "Hello!",
            recipient_agent=InvalidAgent(),  # Invalid agent
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
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2")

    # Create agency with new structure
    agency = Agency(
        entry_points=[agent1],  # Entry point
        communication_flows=[
            (agent1, agent2),  # agent1 can talk to agent2
        ],
        temperature=0,
    )

    # Verify entry points were set up correctly
    assert len(agency.entry_points) == 1
    assert agency.entry_points[0].name == "agent1"

    # Verify communication flows were set up
    assert "agent1" in agency.threads
    assert "agent2" in agency.threads["agent1"]

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
    agent1 = BasicAgent(name="agent1")
    agent2 = BasicAgent(name="agent2")

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
