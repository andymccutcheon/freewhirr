from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from freewhirr.proxy import create_app
from freewhirr.types import ExhaustionPolicy

from .conftest import (
    FREE_A,
    FREE_B,
    completion_payload,
    make_client,
    mock_completions,
    mock_models,
    tool_completion_payload,
)


def _http(app) -> httpx.AsyncClient:  # type: ignore[no-untyped-def]
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@respx.mock
@pytest.mark.asyncio
async def test_proxy_chat_and_health(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "via proxy"))})
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "freewhirr", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "via proxy"
    assert body["model"] == FREE_A


@respx.mock
@pytest.mark.asyncio
async def test_proxy_lists_models(tmp_path: Path) -> None:
    mock_models()
    app = create_app(
        make_client(
            tmp_path,
            exhaustion_policy=ExhaustionPolicy.PAID,
            paid_models=["openai/gpt-4o-mini"],
        )
    )
    async with _http(app) as client:
        data = (await client.get("/v1/models")).json()["data"]
    ids = [item["id"] for item in data]
    assert ids[0] == "freewhirr"
    assert FREE_A in ids
    assert "openai/gpt-4o-mini" in ids


@respx.mock
@pytest.mark.asyncio
async def test_proxy_stream(tmp_path: Path) -> None:
    mock_models()
    body = 'data: {"id":"s1","choices":[{"delta":{"content":"tok"},"index":0}]}\n\ndata: [DONE]\n\n'
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
    async with (
        _http(app) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        text = "".join([part async for part in response.aiter_text()])
    assert "tok" in text
    assert "[DONE]" in text


@respx.mock
@pytest.mark.asyncio
async def test_proxy_complexity_header(tmp_path: Path) -> None:
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
    async with _http(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 402
    assert "high-complexity" in response.json()["detail"]["message"]


@respx.mock
@pytest.mark.asyncio
async def test_proxy_anthropic_messages(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "bonjour"))})
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 32,
                "system": "Be brief.",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"anthropic-version": "2023-06-01", "x-api-key": "unused"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0] == {"type": "text", "text": "bonjour"}


@respx.mock
@pytest.mark.asyncio
async def test_proxy_anthropic_tools(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_B: httpx.Response(200, json=tool_completion_payload(FREE_B))})
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        response = await client.post(
            "/v1/messages",
            json={
                "model": "freewhirr",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "lookup x"}],
                "tools": [
                    {
                        "name": "lookup",
                        "description": "Look up",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                        },
                    }
                ],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0]["type"] == "tool_use"
    assert body["content"][0]["name"] == "lookup"
    posted = [
        json.loads(call.request.content)
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
    ]
    assert posted and posted[0]["model"] == FREE_B
    assert posted[0]["tools"][0]["function"]["name"] == "lookup"


@respx.mock
@pytest.mark.asyncio
async def test_proxy_anthropic_stream(tmp_path: Path) -> None:
    mock_models()
    body = (
        'data: {"id":"s1","choices":[{"delta":{"content":"tok"},"index":0}]}\n\n'
        'data: {"id":"s1","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
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
    async with (
        _http(app) as client,
        client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "freewhirr",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        text = "".join([part async for part in response.aiter_text()])
    assert "event: message_start" in text
    assert "text_delta" in text
    assert "tok" in text
    assert "event: message_stop" in text


@respx.mock
@pytest.mark.asyncio
async def test_proxy_count_tokens(tmp_path: Path) -> None:
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        response = await client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "freewhirr",
                "messages": [{"role": "user", "content": "token me"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["input_tokens"] >= 1


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "pong"))})
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "model": "freewhirr",
                "instructions": "Be brief.",
                "input": "hi",
                "max_output_tokens": 32,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "pong"
    assert body["usage"]["output_tokens"] >= 1


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses_tools_and_followup(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_B: [
                httpx.Response(200, json=tool_completion_payload(FREE_B)),
                httpx.Response(200, json=completion_payload(FREE_B, "done")),
            ]
        }
    )
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        first = await client.post(
            "/v1/responses",
            json={
                "model": "freewhirr",
                "input": "lookup x",
                "tools": [
                    {
                        "type": "function",
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    }
                ],
            },
        )
        assert first.status_code == 200
        payload = first.json()
        assert payload["output"][0]["type"] == "function_call"
        second = await client.post(
            "/v1/responses",
            json={
                "model": "freewhirr",
                "previous_response_id": payload["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": payload["output"][0]["call_id"],
                        "output": "found it",
                    }
                ],
            },
        )
    assert second.status_code == 200
    assert second.json()["output"][0]["content"][0]["text"] == "done"


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses_stream(tmp_path: Path) -> None:
    mock_models()
    body = (
        'data: {"id":"s1","choices":[{"delta":{"content":"tok"},"index":0}]}\n\n'
        'data: {"id":"s1","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
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
    async with (
        _http(app) as client,
        client.stream(
            "POST",
            "/v1/responses",
            json={"model": "freewhirr", "input": "hi", "stream": True},
        ) as response,
    ):
        assert response.status_code == 200
        text = "".join([part async for part in response.aiter_text()])
    assert "event: response.created" in text
    assert "response.output_text.delta" in text
    assert "tok" in text
    assert "event: response.completed" in text
