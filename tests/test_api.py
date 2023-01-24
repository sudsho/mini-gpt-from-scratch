"""tests for the FastAPI inference service."""

import os

import pytest
from fastapi.testclient import TestClient

# import the app object - the startup hook tries to load a checkpoint, but
# if it doesn't exist the app still starts and /health returns model_loaded=false.
os.environ.setdefault("MINI_GPT_CKPT", "out/does-not-exist.pt")

from src.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
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
