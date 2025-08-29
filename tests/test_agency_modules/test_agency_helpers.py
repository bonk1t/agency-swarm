from agency_swarm import Agency, Agent
from agency_swarm.agency.helpers import run_fastapi as helpers_run_fastapi


def test_run_fastapi_creates_new_agency_instance(mocker):
    agent = Agent(name="HelperAgent", instructions="test", model="gpt-4.1")
    agency = Agency(agent)

    captured = {}

    def fake_run_fastapi(*, agencies=None, **kwargs):
        captured["factory"] = agencies["agency"]
        return None

    mocker.patch("agency_swarm.integrations.fastapi.run_fastapi", side_effect=fake_run_fastapi)

    helpers_run_fastapi(agency)

    factory = captured["factory"]
    load_called = False

    def load_cb():
        nonlocal load_called
        load_called = True
        return []

    new_agency = factory(load_threads_callback=load_cb)

    assert load_called, "load_threads_callback was not invoked"
    assert new_agency is not agency, "Factory should create a new Agency instance"
