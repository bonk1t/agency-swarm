"""A minimal example demonstrating basic Agency functionality with a single agent."""

from agency_swarm import Agency, Agent


class SimpleAgent(Agent):
    def __init__(self):
        # Initialize with required parameters
        super().__init__(
            name="SimpleAgent",
            description="A simple agent that can respond to messages",
            instructions="You are a helpful assistant that responds to messages.",
            model="gpt-4o",
            temperature=0.3,
            response_format={"type": "output_text"},  # Use proper response format
        )


async def main():
    # Create the agent
    agent = SimpleAgent()

    # Create the agency with a single entry point
    agency = Agency(entry_points=[agent])

    try:
        # Send a message through the agency
        response = await agency.get_response(
            "Hello! How are you?",  # Simplified message format
            recipient_agent=agent,
        )
        print(
            "Response content:", response.content
        )  # Access content from AgentResponse
        print("Response sender:", response.sender_name)  # Access sender name
    except Exception as e:
        print("Error:", e)
        if hasattr(e, "response"):
            print("Response details:", e.response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
