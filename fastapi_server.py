from agents import ModelSettings

from agency_swarm import Agency, Agent
from agency_swarm.integrations.fastapi import run_fastapi

agent2 = Agent(
    name="Agent2",
    description="Has access to user's username",
    instructions="User's username is dev_user_6123",
    temperature=0.3,
)

agent1 = Agent(
    name="TestAgent",
    instructions="You are a helpful assistant that can answer questions and help with tasks.",
    description="Can call agents 2 and 3",
    model_settings=ModelSettings(
        temperature=0.3,
        parallel_tool_calls=False,
        # max_tokens=16,
    ),
)


agent3 = Agent(
    name="Agent3",
    description="Has access to test tool 2",
    instructions="Take the user name given by TestAgent and use it when you call test tool 2",
    temperature=0,
)

agency1 = Agency(agent1, name="agency1")
agency2 = Agency(agent2, name="agency2")

app = run_fastapi(
    agencies={"agency1": agency1},
    port=8080,
    return_app=False,
    enable_agui=False,
    app_token_env="",
    enable_logging=True
)
