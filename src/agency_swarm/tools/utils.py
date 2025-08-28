from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal, Optional, Union

import httpx
from agents.exceptions import ModelBehaviorError
from agents.run_context import RunContextWrapper
from datamodel_code_generator import DataModelType, PythonVersion
from datamodel_code_generator.model import get_data_model_types
from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def validate_openapi_spec(spec: str):
    spec_dict = json.loads(spec)

    # Validate that 'paths' is present in the spec
    if "paths" not in spec_dict:
        raise ValueError("The spec must contain 'paths'.")

    paths = spec_dict["paths"]
    if not isinstance(paths, dict):
        raise ValueError("The 'paths' field must be a dictionary.")

    for path, path_item in paths.items():
        # Check that each path item is a dictionary
        if not isinstance(path_item, dict):
            raise ValueError(f"Path item for '{path}' must be a dictionary.")

        for operation in path_item.values():
            # Basic validation for each operation
            if "operationId" not in operation:
                raise ValueError("Each operation must contain an 'operationId'.")
            if "description" not in operation:
                raise ValueError("Each operation must contain a 'description'.")

    return spec_dict


def generate_model_from_schema(schema: dict, class_name: str, strict: bool) -> type:
    data_model_types = get_data_model_types(
        DataModelType.PydanticV2BaseModel,
        target_python_version=PythonVersion.PY_310,
    )
    parser = JsonSchemaParser(
        json.dumps(schema),
        data_model_type=data_model_types.data_model,
        data_model_root_type=data_model_types.root_model,
        data_model_field_type=data_model_types.field_model,
        data_type_manager_type=data_model_types.data_type_manager,
        dump_resolve_reference_action=data_model_types.dump_resolve_reference_action,
        use_schema_description=True,
        validation=False,
        class_name=class_name,
        strip_default_none=strict,
    )
    result = parser.parse()
    imports_str = "from typing import List, Dict, Any, Optional, Union, Set, Tuple, Literal\nfrom enum import Enum\n"
    if isinstance(result, str):
        result = imports_str + result
    else:
        result = imports_str + str(result)
    result = result.replace("from __future__ import annotations\n", "")
    result += f"\n\n{class_name}.model_rebuild(force=True)"
    exec_globals = {
        "List": list,
        "Dict": dict,
        "Type": type,
        "Union": Union,
        "Optional": Optional,
        "datetime": datetime,
        "date": date,
        "Set": set,
        "Tuple": tuple,
        "Any": Any,
        "Callable": Callable,
        "Decimal": Decimal,
        "Literal": Literal,
        "Enum": Enum,
    }
    exec(result, exec_globals)
    model = exec_globals.get(class_name)
    if not model:
        raise ValueError(f"Could not extract model from schema {class_name}")
    if hasattr(model, "model_rebuild"):
        try:
            model.model_rebuild(force=True)
        except Exception as e:
            print(f"Warning: Could not rebuild model {class_name} after exec: {e}")
    return model  # type: ignore[return-value]


def create_invoke_for_path(
    path, verb, openapi, param_model, request_body_model, headers=None, params=None, timeout=90
):
    """
    Creates a callback function for a specific path and method.
    This is a factory function that captures the current values of path and method.

    Parameters:
        path: The path to create the callback for.
        verb: The HTTP method to use.
        openapi: The OpenAPI specification.
        param_model: Pydantic model for validating URL/query parameters.
        request_body_model: Pydantic model for validating request body payload.
        headers: Headers to include in the request.
        params: Additional parameters to include in the request.
        timeout: HTTP timeout in seconds.

    Returns:
        An async callback function that makes the appropriate HTTP request.
    """
    fixed_params = params or {}

    async def _invoke(
        ctx: RunContextWrapper[Any],
        input: str,
        *,
        verb_: str = verb,
        path_: str = path,
        param_model_: type[BaseModel] = param_model,
        request_body_model_: type[BaseModel] = request_body_model,
    ):
        """Actual HTTP call executed by the agent."""
        payload = json.loads(input) if input else {}

        # split out parts for old-style structure
        param_container: dict[str, Any] = payload.get("parameters", {})

        if param_model_:
            # Validate parameters
            try:
                parsed = param_model_(**param_container) if param_container else param_model_()
                param_container = parsed.model_dump()
            except ValidationError as e:
                raise ModelBehaviorError(
                    f"Invalid JSON input in parameters for tool {param_model_.__name__}: {e}"
                ) from e

        body_payload = payload.get("requestBody")

        if request_body_model_:
            # Validate request body
            try:
                parsed = request_body_model_(**body_payload) if body_payload else request_body_model_()
                body_payload = parsed.model_dump()
            except ValidationError as e:
                raise ModelBehaviorError(
                    f"Invalid JSON input in request body for tool {request_body_model_.__name__}: {e}"
                ) from e

        url = f"{openapi['servers'][0]['url']}{path_}"
        for key, val in param_container.items():
            token = f"{{{key}}}"
            if token in url:
                url = url.replace(token, str(val))
                # null-out so it doesn't go into query string
                param_container[key] = None
        url = url.rstrip("/")

        query_params = {k: v for k, v in param_container.items() if v is not None}
        if fixed_params:
            query_params = {**query_params, **fixed_params}

        json_body = body_payload if verb_.lower() in {"post", "put", "patch", "delete"} else None

        logger.info(f"Calling URL: {url}\nQuery Params: {query_params}\nJSON Body: {json_body}")

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                verb_.upper(),
                url,
                params=query_params,
                json=json_body,
                headers=headers,
            )
            try:
                logger.info(f"Response from {url}: {resp.json()}")
                return resp.json()
            except Exception:
                return resp.text

    return _invoke
