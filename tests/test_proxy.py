from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from freeloader.proxy import create_app
from freeloader.types import ExhaustionPolicy

from .conftest import FREE_A, FREE_B, completion_payload, make_client, mock_completions, mock_models


@respx.mock
def test_proxy_chat_and_health(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "via proxy"))})
    app = create_app(make_client(tmp_path))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post(
        "/v1/chat/completions",
        json={"model": "freeloader", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "via proxy"
    assert body["model"] == FREE_A


@respx.mock
def test_proxy_lists_models(tmp_path: Path) -> None:
    mock_models()
    app = create_app(
        make_client(
            tmp_path,
            exhaustion_policy=ExhaustionPolicy.PAID,
            paid_models=["openai/gpt-4o-mini"],
        )
    )
    client = TestClient(app)
    data = client.get("/v1/models").json()["data"]
    ids = [item["id"] for item in data]
    assert ids[0] == "freeloader"
    assert FREE_A in ids
    assert "openai/gpt-4o-mini" in ids


@respx.mock
def test_proxy_stream(tmp_path: Path) -> None:
    mock_models()
    body = (
        'data: {"id":"s1","choices":[{"delta":{"content":"tok"},"index":0}]}\n\n'
        "data: [DONE]\n\n"
    )
    mock_completions(
        {
            FREE_A: httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        }
    )
    app = create_app(make_client(tmp_path))
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "gpt-4o-mini", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    assert "tok" in text
    assert "[DONE]" in text


@respx.mock
def test_proxy_complexity_header(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(500, json={"error": {"message": "x"}}),
            FREE_B: httpx.Response(500, json={"error": {"message": "x"}}),
        }
    )
    app = create_app(
        make_client(
            tmp_path,
            exhaustion_policy="paid",
            paid_models=["openai/gpt-4o-mini"],
            paid_only_high_complexity=True,
        )
    )
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 402
    assert "high-complexity" in response.json()["detail"]["message"]
