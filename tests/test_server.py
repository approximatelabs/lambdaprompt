import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from lambdaprompt.server import main

    def get_settings_override():
        return main.Settings(sqlite_path="sqlite:///")

    main.app.dependency_overrides[main.get_settings] = get_settings_override

    return TestClient(main.app)


def test_app(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
