"""tests for the FastAPI inference service."""

import os

import pytest
from fastapi.testclient import TestClient

# point at a non-existent ckpt BEFORE importing the app, so the startup hook
# is a no-op and tests don't depend on any trained checkpoint being present.
os.environ["MINI_GPT_CKPT"] = "out/__pytest_does_not_exist__.pt"

from src.api.main import app, _state  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # reset the module-global state so order of tests doesn't matter
    _state["model"] = None
    _state["config"] = None
    _state["tokenizer"] = None
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # no checkpoint -> model_loaded should be False
    assert body["model_loaded"] is False


def test_generate_503_when_no_checkpoint(client):
    r = client.post(
        "/generate",
        json={"prompt": "hi", "max_new_tokens": 5, "temperature": 0.8, "top_k": 40},
    )
    assert r.status_code == 503


def test_generate_validates_input(client):
    # empty prompt should fail pydantic validation
    r = client.post("/generate", json={"prompt": ""})
    assert r.status_code == 422

    # negative tokens should fail
    r = client.post("/generate", json={"prompt": "a", "max_new_tokens": -1})
    assert r.status_code == 422

    # temperature out of range
    r = client.post("/generate", json={"prompt": "a", "temperature": 0.0})
    assert r.status_code == 422
