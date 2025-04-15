# Agency Swarm Framework Plan

## Overview

This plan describes the current state and final design for the Agency Swarm Framework as we complete the migration from OpenAI's Assistants API to the newer Responses API. The emphasis is on streamlining thread management by removing legacy `main_thread` references and unifying all message routing through a flat, consistent structure.

## Communication Model

- **Valid Communication Flows:**
  - **User → Agent:** Users send messages to any designated entry point Agent.
  - **Agent → Agent:** Agents can send messages to one another along pre-defined directional flows.

- **Invalid Communication Flows:**
  - Agents cannot be the recipient of a User-to-User or Agent-to-User message.

- **Recipient Constraints:**
  - Only Agents can be recipients; Users are strictly senders.

## Dependencies

- **OpenAI Responses API:** Used for agent responses and tool execution.
- **Pydantic:** For data validation and settings management.
- **Python Version:** 3.9+ (to support async and advanced type hints)

## File Structure

- **Core Components:**
  - `/agency_swarm/agents/agent.py` — Base Agent class.
  - `/agency_swarm/threads/thread.py` — Thread management including message history and routing.
  - `/agency_swarm/tools/` — Base tool classes and implementations.
- **Testing:**
  - `/tests/test_agency.py` — Tests for agency communication flows.
  - `/tests/test_message_schema.py` — Validates message schema compliance.

## Thread Management

- **Unified Storage:**
  - All communication threads are stored in a flat dictionary with keys formatted as `"sender->recipient"` (e.g., `"user->AgentName"`, `"AgentA->AgentB"`).

- **Elimination of Legacy `main_thread`:**
  - All special handling for `main_thread` has been removed.
  - State management (loading and saving) and response streaming now iterate over the uniform thread dictionary without any hardcoded keys.

- **Thread Initialization:**
  - **User → Agent:** A thread is created for each conversation between a User and every entry point Agent.
  - **Agent → Agent:** Threads are established according to the specified communication flows.

- **State Persistence:**
  - Conversation histories are saved and loaded by iterating over all thread keys in the flat dictionary, ensuring consistency and reducing complexity.

## Message Format

- **Input Messages:**
  ```python
  {
      "role": "user" | "assistant" | "tool",
      "content": [
          {"type": "input_text", "text": str} |
          {"type": "input_file", "file_id": str} |
          {"type": "output_text", "text": str}
      ]
  }

	•	Tool Calls:

{
    "type": "function_call",
    "id": str,
    "call_id": str,
    "name": str,
    "arguments": str  # JSON string
}


	•	Tool Results:

{
    "role": "tool",
    "type": "function_call_output",
    "call_id": str,
    "content": str
}



Tool and Function Call Handling
	•	Execution:
	•	Tools must receive their arguments as JSON strings compliant with the Responses API.
	•	Schema Compliance:
	•	Every tool call and its result are validated against the ResponseFunctionToolCall schema.
	•	History Tracking:
	•	All tool executions and responses are appended to the conversation history in the respective thread.

State Management
	•	Conversation History:
	•	Complete message histories are maintained in each dedicated thread.
	•	Persistence:
	•	State is saved and loaded by iterating over the flat thread dictionary without any special cases.
	•	Validation:
	•	Each message’s format is validated to ensure strict adherence to the Responses API standards.

Implementation Steps
	1.	Update Thread Structure:
	•	Refactor initialization to use flat thread keys in the format "sender->recipient".
	•	Remove all legacy references to main_thread (in state management, streaming, and demos).
	•	Revise the _get_thread() method to dynamically return the correct thread based on sender and recipient.
	2.	Refactor State Persistence and Response Streaming:
	•	Update _load_state() and _save_state() to iterate over the flat dictionary of threads.
	•	Refactor get_response_stream() to use _get_thread() instead of accessing a hardcoded thread key.
	3.	Adjust Demo and UI Components:
	•	Update demo interfaces (e.g., Gradio and terminal demos) to reflect the new thread lookup logic.
	•	Modify file upload and tool invocation flows to use the thread associated with the recipient Agent.
	4.	Tool Execution and Message Format Compliance:
	•	Align all tool call formats with the Responses API, including proper JSON schema validation.
	•	Ensure that all function call structures include the necessary call_id and status fields.
	5.	Testing and Verification:
	•	Run all tests covering user-to-agent, agent-to-agent communication, tool execution, and state persistence.
	•	Validate that message histories, streaming responses, and file handling work as expected under the new unified thread system.

Testing Strategy
	•	Core Communication Flows:
	•	Validate User → Agent flows.
	•	Confirm Agent → Agent interactions.
	•	Edge Cases:
	•	Test error handling and validation for improper message formats.
	•	Check state persistence after various interactions.
	•	Verify response streaming operates correctly without legacy main_thread references.
	•	UI and Demo Functionality:
	•	Ensure demos (Gradio and terminal) correctly reflect the new thread management.
	•	Regression Testing:
	•	Maintain backward compatibility on public APIs while enforcing the updated thread management logic.

Final Objectives
	•	Unified Thread Management:
Develop a clean and consistent communication model using a flat thread dictionary, completely removing legacy main_thread references.
	•	API Compliance:
Ensure all message and tool execution formats conform to the Responses API specifications.
	•	Robustness and Clarity:
Simplify state management and response streaming while preserving backward compatibility and full functionality.
	•	Comprehensive Testing:
Achieve full test coverage for all communication flows, state persistence, and UI interactions.


⸻

Analysis of the Thread Management System (can be removed after fixes are implemented)

The current implementation has several key issues that need addressing to complete the refactoring:
	1.	Legacy “main_thread” References:
Several parts of the code (state loading/saving, response streaming, and even file handling) still reference a special main_thread. This is a remnant from the previous design. The new requirements call for a unified, flat thread dictionary keyed solely by "sender->recipient". Every thread should follow the same lookup and state persistence rules with no special case for a main thread.
	2.	Inconsistent Thread Initialization:
The _init_threads() method only initializes threads for user-to-entry-point and agent-to-agent communication based on the communication flows. While that is good, it still coexists with outdated handling (like checking for "main_thread"). The plan is to standardize this so that each communication path—whether User→Agent or Agent→Agent—is created using a consistent key format.
	3.	State Management:
Both _load_state() and _save_state() include explicit handling for main_thread, which creates unnecessary complexity. This needs to be replaced with iteration over the new flat thread dictionary, ensuring that all messages are persisted in the same manner.
	4.	Response Streaming and Tool Execution:
The method get_response_stream() directly accesses self.threads["main_thread"]. Instead, it should retrieve the appropriate thread via _get_thread(), ensuring that the lookup is done based on the recipient rather than a hardcoded key.
	5.	Simplification and Type Consistency:
With Agents guaranteed as recipients (by type validation on the Thread constructor), any legacy checks or comments about handling User objects as recipients are obsolete. We can remove these checks for clarity and enforce the model—only Users can initiate conversations, never receive them.

⸻
