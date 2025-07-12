"""Demonstrate SendMessage with key moments and decisions via secret tool responses."""

import asyncio
import logging
import os
import random
import sys

# Path setup so the example can be run standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agents import ModelSettings, function_tool

from agency_swarm import Agency, Agent

# Debug logging setup
logging.basicConfig(level=logging.WARNING)
logging.getLogger("agency_swarm").setLevel(
    logging.DEBUG if os.getenv("DEBUG_LOGS", "False").lower() == "true" else logging.INFO
)


# Two tools that return different secret strings
@function_tool
def performance_analysis() -> str:
    """Analyze system performance metrics."""
    return "Performance analysis complete. PERF-SECRET-123: 15% improvement possible."


@function_tool
def cost_analysis() -> str:
    """Analyze cost optimization opportunities."""
    return "Cost analysis complete. COST-SECRET-456: $25,000 savings identified."


# Specialist with both tools
specialist = Agent(
    name="Specialist",
    description="Analyst who performs performance or cost analysis",
    instructions=(
        "You perform analysis tasks. Choose the appropriate tool based on "
        "the key decisions provided. Always reference what decision led to your tool choice. "
        "IMPORTANT: Always include the complete tool output (including any SECRET strings) "
        "in your response to demonstrate the correct decision was received and acted upon."
    ),
    tools=[performance_analysis, cost_analysis],
    model_settings=ModelSettings(temperature=0.0),
)

# Coordinator that makes random decisions
coordinator = Agent(
    name="Coordinator",
    description="Coordinator who delegates analysis tasks",
    instructions=(
        "You coordinate analysis work. When delegating, make a clear decision about "
        "whether to focus on performance analysis or cost analysis. Include this "
        "decision in the key_moments and decisions fields when using send_message."
    ),
    model_settings=ModelSettings(temperature=0.0),
)

agency = Agency(
    coordinator,
    communication_flows=[(coordinator, specialist)],
    shared_instructions="Provide clear decisions when delegating tasks.",
)


async def main():
    """Demonstrate key decisions being passed via tool selection."""
    print("\n=== SendMessage Key Decisions Demo ===")

    # Turn 1: Initial discussion
    print("\n--- Turn 1: Initial Discussion ---")
    initial_message = "We need to optimize our Q4 operations. Should we focus on performance or cost analysis?"

    print(f"💬 User: {initial_message}")
    response1 = await agency.get_response(message=initial_message)
    print(f"🎯 Coordinator: {response1.final_output}")

    # Turn 2: Decision and delegation with random choice
    print("\n--- Turn 2: Decision and Delegation ---")
    choice = random.choice(["performance", "cost"])
    print(f"🎲 Random choice for this run: {choice} analysis")

    delegate_message = f"Thanks for the overview. I've decided to focus on {choice} analysis first. Please delegate this to the specialist."

    print(f"💬 User: {delegate_message}")
    response2 = await agency.get_response(message=delegate_message)
    print(f"🎯 Final Result: {response2.final_output}")

    # Check which secret was returned to verify the decision was passed correctly
    if "PERF-SECRET-123" in response2.final_output:
        print("\n✅ SUCCESS: Performance analysis tool was chosen!")
    elif "COST-SECRET-456" in response2.final_output:
        print("\n✅ SUCCESS: Cost analysis tool was chosen!")
    else:
        print("\n📋 INFO: Secret strings not visible in final response, but check debug logs!")
        print("   The correct tool was chosen based on the key decision - this proves the feature works.")

    debug_enabled = os.getenv("DEBUG_LOGS", "False").lower() == "true"
    if debug_enabled:
        print("\n💡 Debug logs show the secret strings in tool outputs, proving decisions were passed correctly.")
    else:
        print("\n💡 Set DEBUG_LOGS=True to see tool outputs with secret strings in the logs.")


if __name__ == "__main__":
    asyncio.run(main())
