"""Tests for Agency Swarm core functionality.

These tests verify critical functionality of the Agency Swarm framework:
1. Basic agent responses and message handling
2. State persistence through complete conversation history
3. Agent uniqueness and proper initialization
4. Thread management and message routing
"""

import asyncio
import inspect
import json
import logging
import os
import shutil
import time
import unittest
from typing import Type

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing_extensions import override

from agency_swarm import Agency
from agency_swarm.agents import Agent
from agency_swarm.messages import AgentResponse, Text
from agency_swarm.threads import AgencyEventHandler, Thread
from agency_swarm.tools import BaseTool
from agency_swarm.tools.tool_factory import ToolFactory

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


class TestAgency(unittest.TestCase):
    """Test suite for Agency functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.loaded_agents_settings = []
        cls.num_schemas = 0
        cls.agency = None

    @pytest.mark.asyncio
    async def test_new_agency_structure(self):
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

        self.check_all_agents_settings(True)

    def test_08_async_agent_communication(self):
        """it should communicate between agents asynchronously"""
        self.__class__.agency.get_completion(
            "Please tell TestAgent2 hello.",
            tool_choice={"type": "function", "function": {"name": "SendMessage"}},
            recipient_agent=self.__class__.agent1,
        )

        time.sleep(10)

        num_on_all_streams_end_calls = 0
        delta_value = ""
        full_text = ""

        class EventHandler(AgencyEventHandler):
            @override
            def on_text_delta(self, delta, snapshot):
                nonlocal delta_value
                delta_value += delta.value

            @override
            def on_text_done(self, text: Text) -> None:
                nonlocal full_text
                full_text += text.value

            @override
            @classmethod
            def on_all_streams_end(cls):
                nonlocal num_on_all_streams_end_calls
                num_on_all_streams_end_calls += 1

        message = self.__class__.agency.get_completion_stream(
            "Please check response. If output includes `TestAgent2's Response`, say 'success'. If the function output does not include `TestAgent2's Response`, or if you get a System Notification, or an error instead, say 'error'.",
            tool_choice={"type": "function", "function": {"name": "GetResponse"}},
            recipient_agent=self.__class__.agent1,
            event_handler=EventHandler,
        )

        self.assertTrue(num_on_all_streams_end_calls == 1)

        self.assertTrue(delta_value == full_text == message)

        self.assertTrue(EventHandler.agent_name == "User")
        self.assertTrue(EventHandler.recipient_agent_name == "TestAgent1")

        if "error" in message.lower():
            self.assertFalse(
                "error" in message.lower(), self.__class__.agency.main_thread.thread_url
            )

        self.assertTrue(self.__class__.agency.main_thread.id)
        self.assertTrue(
            self.__class__.agency.agents_and_threads["TestAgent1"]["TestAgent2"].id
        )

        for agent in self.__class__.agency.agents:
            self.assertTrue(
                agent.id
                in [
                    settings["id"] for settings in self.__class__.loaded_agents_settings
                ]
            )

    def test_09_async_tool_calls(self):
        """it should execute tools asynchronously"""

        class PrintTool(BaseTool):
            class ToolConfig:
                async_mode = "threading"

            def run(self, **kwargs):
                time.sleep(2)  # Simulate a delay
                return "Printed successfully."

        class AnotherPrintTool(BaseTool):
            class ToolConfig:
                async_mode = "threading"

            def run(self, **kwargs):
                time.sleep(2)  # Simulate a delay
                return "Another print successful."

        ceo = Agent(name="CEO", tools=[PrintTool, AnotherPrintTool])

        agency = Agency([ceo], temperature=0)

        result = agency.get_completion(
            "Use 2 print tools together at the same time and output the results exectly as they are. ",
            yield_messages=False,
        )

        self.assertIn("success", result.lower(), agency.main_thread.thread_url)
        self.assertIn("success", result.lower(), agency.main_thread.thread_url)

    def test_10_concurrent_API_calls(self):
        """it should execute API calls concurrently with asyncio"""

        # Create a mock client that will be used instead of httpx
        class MockClient:
            def __init__(self, **kwargs):
                self.timeout = kwargs.get("timeout", None)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

            async def get(self, url, params=None, headers=None):
                # Verify that the domain parameter is correctly set in the URL
                if "print-headers-gntxktyfsq-uc.a.run.app" in url:

                    class MockResponse:
                        def json(self):
                            return {"headers": {"test": "success"}}

                        def raise_for_status(self):
                            pass

                    return MockResponse()
                raise ValueError(f"Invalid URL: {url}")

            async def aclose(self):
                pass

        # Patch httpx.AsyncClient with our mock
        original_client = httpx.AsyncClient
        httpx.AsyncClient = MockClient

        try:
            tools = []
            with open("./data/schemas/get-headers-params.json", "r") as f:
                tools = ToolFactory.from_openapi_schema(f.read(), {})

            ceo = Agent(
                name="CEO",
                tools=tools,
                instructions="""You are an agent that tests concurrent API calls. You must say 'success' if the output contains headers, and 'error' if it does not and **nothing else**.""",
            )

            agency = Agency([ceo], temperature=0)

            result = agency.get_completion(
                "Please call PrintHeaders tool TWICE at the same time in a single message with domain='print-headers' and query='test'. If any of the function outputs do not contain headers, please say 'error'."
            )

            self.assertTrue(
                result.lower().count("error") == 0, agency.main_thread.thread_url
            )
        finally:
            # Restore original client
            httpx.AsyncClient = original_client

    def test_11_structured_outputs(self):
        class MathReasoning(BaseModel):
            class Step(BaseModel):
                explanation: str
                output: str

            steps: list[Step]
            final_answer: str

        math_tutor_prompt = """
            You are a helpful math tutor. You will be provided with a math problem,
            and your goal will be to output a step by step solution, along with a final answer.
            For each step, just provide the output as an equation use the explanation field to detail the reasoning.
        """

        agent = Agent(
            name="MathTutor",
            response_format=MathReasoning,
            instructions=math_tutor_prompt,
        )

        agency = Agency([agent], temperature=0)

        result = agency.get_completion("how can I solve 8x + 7 = -23")

        # check if result is a MathReasoning object
        self.assertTrue(MathReasoning.model_validate_json(result))

        result = agency.get_completion_parse(
            "how can I solve 3x + 2 = 14", response_format=MathReasoning
        )

        # check if result is a MathReasoning object
        self.assertTrue(isinstance(result, MathReasoning))

    # --- Helper methods ---

    def get_class_folder_path(self):
        return os.path.abspath(os.path.dirname(inspect.getfile(self.__class__)))

    def check_agent_settings(self, agent, async_mode=False):
        try:
            settings_path = agent.get_settings_path()
            self.assertTrue(os.path.exists(settings_path))
            with open(settings_path, "r") as f:
                settings = json.load(f)
                for assistant_settings in settings:
                    if assistant_settings["id"] == agent.id:
                        self.assertTrue(
                            agent._check_parameters(assistant_settings, debug=True)
                        )

            assistant = agent.assistant
            self.assertTrue(assistant)
            self.assertTrue(agent._check_parameters(assistant.model_dump(), debug=True))
            if agent.name == "TestAgent1":
                num_tools = 3 if not async_mode else 4

                self.assertTrue(
                    len(
                        assistant.tool_resources.model_dump()["code_interpreter"][
                            "file_ids"
                        ]
                    )
                    == 3
                )
                self.assertTrue(
                    len(
                        assistant.tool_resources.model_dump()["file_search"][
                            "vector_store_ids"
                        ]
                    )
                    == 1
                )

                vector_store_id = assistant.tool_resources.model_dump()["file_search"][
                    "vector_store_ids"
                ][0]
                vector_store_files = agent.client.vector_stores.files.list(
                    vector_store_id=vector_store_id
                )

                file_ids = [file.id for file in vector_store_files.data]

                # Add debug output
                print("Vector store files:", len(file_ids))

                self.assertTrue(len(file_ids) == 8)
                # check retrieval tools is there
                self.assertTrue(len(assistant.tools) == num_tools)
                self.assertTrue(len(agent.tools) == num_tools)
                self.assertTrue(assistant.tools[0].type == "code_interpreter")
                self.assertTrue(assistant.tools[1].type == "file_search")
                if not async_mode:
                    self.assertTrue(
                        assistant.tools[1].file_search.max_num_results == 49
                    )  # Updated line
                self.assertTrue(assistant.tools[2].type == "function")
                self.assertTrue(assistant.tools[2].function.name == "SendMessage")
                self.assertFalse(assistant.tools[2].function.strict)
                if async_mode:
                    self.assertTrue(assistant.tools[3].type == "function")
                    self.assertTrue(assistant.tools[3].function.name == "GetResponse")
                    self.assertFalse(assistant.tools[3].function.strict)

            elif agent.name == "TestAgent2":
                self.assertTrue(len(assistant.tools) == self.__class__.num_schemas + 1)
                for tool in assistant.tools:
                    self.assertTrue(tool.type == "function")
                    self.assertTrue(
                        tool.function.name in [tool.__name__ for tool in agent.tools]
                    )
                test_tool = next(
                    (
                        tool
                        for tool in assistant.tools
                        if tool.function.name == "TestTool"
                    ),
                    None,
                )
                self.assertTrue(test_tool.function.strict, test_tool)
            elif agent.name == "CEO":
                num_tools = 1 if not async_mode else 2
                self.assertFalse(assistant.tool_resources.code_interpreter)
                self.assertFalse(assistant.tool_resources.file_search)
                self.assertTrue(len(assistant.tools) == num_tools)
            else:
                pass
        except Exception as e:
            print("Error checking agent settings ", agent.name)
            raise e

    def check_all_agents_settings(self, async_mode=False):
        self.check_agent_settings(self.__class__.ceo, async_mode=async_mode)
        self.check_agent_settings(self.__class__.agent1, async_mode=async_mode)
        self.check_agent_settings(self.__class__.agent2, async_mode=async_mode)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./test_agents")
        # os.remove("./settings.json")
        if cls.agency:
            cls.agency.delete()


if __name__ == "__main__":
    unittest.main()
