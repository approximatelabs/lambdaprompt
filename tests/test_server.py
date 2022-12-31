import os
from typing import List

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(mocker):
    from lambdaprompt.server import main

    class newSettings(main.Settings):
        sqlite_path = ":memory:"
        prompt_library_paths: List[str] = [
            str(os.path.join(os.path.split(__file__)[0], "library"))
        ]

    mocker.patch("lambdaprompt.server.main.get_settings", return_value=newSettings())

    with TestClient(main.app) as client:
        yield client


def test_list_prompts(client):
    response = client.get("/list_prompts")
    assert response.status_code == 200
    result = response.json()
    assert len(result["prompts"]) == 2


def test_direct_call_greet(client):
    response = client.get("/prompt/greet_name", params={"name": "test"})
    assert response.status_code == 200
    result = response.json()
    assert result == "Hello test"


def test_direct_call_async_sleep_greet(client):
    response = client.get(
        "/prompt/delayed_greet", params={"name": "test", "sleeptime": 0.1}
    )
    assert response.status_code == 200
    result = response.json()
    assert result == "Hello test"


def test_background_call_greet(client):
    response = client.get("/async/greet_name", params={"name": "test"})
    assert response.status_code == 200
    result = response.json()
    assert result["jobid"] is not None
    jobid = result["jobid"]
    response = client.get("/background_task_trace", params={"jobid": jobid})
    assert response.status_code == 200
    result = response.json()
    response = client.get("/background_result", params={"jobid": jobid})
    assert response.status_code == 200
    result = response.json()
    assert result["result"] == "Hello test"


def test_background_call_async_greet(client):
    response = client.get(
        "/async/delayed_greet", params={"name": "test", "sleeptime": 0.1}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["jobid"] is not None
    jobid = result["jobid"]
    response = client.get("/background_task_trace", params={"jobid": jobid})
    assert response.status_code == 200
    result = response.json()
    response = client.get("/background_result", params={"jobid": jobid})
    assert response.status_code == 200
    result = response.json()
    assert result["result"] == "Hello test"
