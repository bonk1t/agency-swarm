import json

import pytest

from agency_swarm.tools.utils import validate_openapi_spec


@pytest.fixture
def base_spec():
    return {"servers": [{"url": "https://api.example.com"}], "paths": {}}

class TestValidateOpenAPISpec:
    @pytest.mark.parametrize(
        "spec,should_pass",
        [
            ({"paths": {"/users": {"get": {"operationId": "getUsers", "description": "Get users"}}}}, True),
            ({"info": {"title": "API"}}, False),  # Missing paths
            ({"paths": {"/users": "invalid"}}, False),  # Invalid path item
            ({"paths": {"/users": {"get": {"description": "Get users"}}}}, False),  # Missing operationId
            ({"paths": {"/users": {"get": {"operationId": "getUsers"}}}}, False),  # Missing description
        ],
    )
    def test_validation(self, spec, should_pass):
        if should_pass:
            result = validate_openapi_spec(json.dumps(spec))
            assert result == spec
        else:
            with pytest.raises(ValueError):
                validate_openapi_spec(json.dumps(spec))
