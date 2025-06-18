import os
from typing import Callable, List, Mapping

from agents.tool import FunctionTool
from dotenv import load_dotenv

from agency_swarm.agency import Agency
from agency_swarm.agent import Agent

load_dotenv()


def run_fastapi(
    agencies: Mapping[str, Callable[..., Agency]] | list[Agency] | None = None,
    tools: list[type[FunctionTool]] | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    app_token_env: str = "APP_TOKEN",
    return_app: bool = False,
    cors_origins: List[str] = ["*"],
):
    """
    Launch a FastAPI server exposing endpoints for multiple agencies and tools.
    ``agencies`` should be a mapping of endpoint names to *factory callables*.
    Each callable must return a new :class:`Agency` instance and should accept an
    optional ``load_threads_callback`` argument. The mapping keys become part of
    the endpoint paths (``/name/get_completion`` and ``/name/get_completion_stream``).

    Tools can be provided as classes and will be served at ``/tool/ToolName``.
    """
    if (agencies is None or len(agencies) == 0) and (tools is None or len(tools) == 0):
        print("No endpoints to deploy. Please provide at least one agency or tool.")
        return

    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        from .fastapi_utils.endpoint_handlers import (
            exception_handler,
            get_verify_token,
            make_response_endpoint,
            make_stream_endpoint,
            make_tool_endpoint,
        )
        from .fastapi_utils.request_models import BaseRequest, add_agent_validator
    except ImportError:
        print("FastAPI deployment dependencies are missing. Please install agency-swarm[fastapi] package")
        return

    app_token = os.getenv(app_token_env)
    if app_token is None or app_token == "":
        print(f"Warning: {app_token_env} is not set. Authentication will be disabled.")
    verify_token = get_verify_token(app_token)

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    endpoints = []
    agency_names = []

    if agencies:
        agency_factories: dict[str, Callable[..., Agency]] = {}

        if isinstance(agencies, Mapping):
            agency_factories.update(agencies)
        else:  # legacy list of instances
            for idx, agency in enumerate(agencies):
                if not isinstance(agency, Agency):
                    raise TypeError("agencies list must contain Agency instances")
                name = getattr(agency, "name", None)
                if name is None:
                    name = "agency" if len(agencies) == 1 else f"agency_{idx + 1}"
                name = name.replace(" ", "_")
                agency_factories[name] = lambda a=agency: a

        for agency_name, factory in agency_factories.items():
            if agency_name in agency_names:
                raise ValueError(
                    f"Agency name {agency_name} is already in use. "
                    "Please provide a unique name in the agency's 'name' parameter."
                )
            agency_names.append(agency_name)

            prototype = factory()
            AGENT_INSTANCES: dict[str, Agent] = dict(prototype.agents.items())

            class VerboseRequest(BaseRequest):
                verbose: bool = False

            AgencyRequest = add_agent_validator(VerboseRequest, AGENT_INSTANCES)
            AgencyRequestStreaming = add_agent_validator(BaseRequest, AGENT_INSTANCES)

            app.add_api_route(
                f"/{agency_name}/get_completion",
                make_response_endpoint(AgencyRequest, factory, verify_token),
                methods=["POST"],
            )
            app.add_api_route(
                f"/{agency_name}/get_completion_stream",
                make_stream_endpoint(AgencyRequestStreaming, factory, verify_token),
                methods=["POST"],
            )
            endpoints.append(f"/{agency_name}/get_completion")
            endpoints.append(f"/{agency_name}/get_completion_stream")

    if tools:
        for tool in tools:
            tool_name = tool.name
            tool_handler = make_tool_endpoint(tool, verify_token)
            app.add_api_route(f"/tool/{tool_name}", tool_handler, methods=["POST"], name=tool_name)
            endpoints.append(f"/tool/{tool_name}")

    app.add_exception_handler(Exception, exception_handler)

    print("Created endpoints:\n" + "\n".join(endpoints))

    if return_app:
        return app

    print(f"Starting FastAPI server at http://{host}:{port}")

    uvicorn.run(app, host=host, port=port)
