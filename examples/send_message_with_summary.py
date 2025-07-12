"""Demonstrate SendMessage with extra context fields."""

import asyncio
import logging
import os
import sys

# Path setup so the example can be run standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agency_swarm import Agent, Agency

logging.basicConfig(level=logging.WARNING)
logging.getLogger("agency_swarm").setLevel(logging.INFO)

# Coordinator agent will delegate work to the Specialist
coordinator = Agent(
    name="Coordinator",
    instructions=(
        "You plan work and delegate to the Specialist agent. "
        "When delegating, use the send_message tool and include summaries "
        "of key moments and decisions so far."
    ),
)

specialist = Agent(
    name="Specialist",
    instructions="You complete tasks given by the Coordinator agent.",
)

agency = Agency(
    coordinator,
    communication_flows=[(coordinator, specialist)],
    shared_instructions="Always provide key moments and decisions when using send_message.",
)


async def main():
    """Demonstrate the SendMessage tool with Key moments and Decisions fields."""
    print("\n--- SendMessage With Summary Demo ---")

    user_message = "Kick off planning for project Phoenix."
    response = await agency.get_response(message=user_message)

    if response and response.final_output:
        print(f"Final Output from {coordinator.name}: {response.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
