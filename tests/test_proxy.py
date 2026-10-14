from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from freewhirr.proxy import create_app
from freewhirr.responses import aopenai_chunks_to_responses_sse
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
async def test_proxy_requested_paid_catalogue_model_uses_free_queue(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "free via proxy")),
            "openai/gpt-4o-mini": httpx.Response(
                200, json=completion_payload("openai/gpt-4o-mini", "paid", cost=0.01)
            ),
        }
    )
    app = create_app(make_client(tmp_path, honor_requested_model=True))
    async with _http(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["model"] == FREE_A
    upstream_models = [
        json.loads(call.request.content)["model"]
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
    ]
    assert upstream_models == [FREE_A]


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
async def test_proxy_complexity_header_orders_strong_free_model_first(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "regular")),
            FREE_B: httpx.Response(200, json=completion_payload(FREE_B, "strong")),
        }
    )
    app = create_app(make_client(tmp_path, strong_free_models=[FREE_B]))
    async with _http(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hard"}]},
            headers={"X-Freewhirr-Complexity": "high"},
        )
    assert response.status_code == 200
    assert response.json()["model"] == FREE_B


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


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses_stream_tool_call_continuation(tmp_path: Path) -> None:
    mock_models()
    tool_stream = (
        'data: {"id":"s1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
        '"type":"function","function":{"name":"lookup","arguments":"{\\"q\\":\\"x\\"}"}}]},'
        '"index":0}]}'
        '\n\ndata: {"id":"s1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
        "\n\ndata: [DONE]\n\n"
    )
    done_stream = (
        'data: {"id":"s2","choices":[{"delta":{"content":"done"},"index":0}]}'
        '\n\ndata: {"id":"s2","choices":[{"delta":{},"finish_reason":"stop"}]}'
        "\n\ndata: [DONE]\n\n"
    )
    mock_completions(
        {
            FREE_B: [
                httpx.Response(
                    200, content=tool_stream.encode(), headers={"content-type": "text/event-stream"}
                ),
                httpx.Response(
                    200, content=done_stream.encode(), headers={"content-type": "text/event-stream"}
                ),
            ]
        }
    )
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        async with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "freewhirr", "input": "lookup x", "stream": True},
        ) as first:
            first_text = "".join([part async for part in first.aiter_text()])
        first_id = _completed_response(first_text)["id"]
        async with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "freewhirr",
                "previous_response_id": first_id,
                "input": [{"type": "function_call_output", "call_id": "call-1", "output": "found"}],
                "stream": True,
            },
        ) as second:
            second_text = "".join([part async for part in second.aiter_text()])
    assert "event: response.completed" in second_text
    sent = _last_completion_request().read()
    payload = json.loads(sent)
    assert payload["messages"][-2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"x"}'},
            }
        ],
    }
    assert payload["messages"][-1]["role"] == "tool"
    assert payload["messages"][-1]["tool_call_id"] == "call-1"


@pytest.mark.asyncio
async def test_responses_stream_remembers_before_final_event_is_yielded() -> None:
    async def chunks():
        yield {"choices": [{"delta": {"content": "hello"}, "index": 0}]}

    remembered: list[dict[str, object]] = []
    stream = aopenai_chunks_to_responses_sse(
        chunks(), model="test", on_complete=lambda message: remembered.append(message)
    )
    async for event in stream:
        if "event: response.completed" in event:
            break
    await stream.aclose()
    assert remembered == [{"role": "assistant", "content": "hello"}]


@pytest.mark.asyncio
async def test_responses_stream_empty_turn_has_empty_content_in_memory() -> None:
    async def chunks():
        if False:
            yield {}

    remembered: list[dict[str, object]] = []
    async for _ in aopenai_chunks_to_responses_sse(
        chunks(), model="test", on_complete=lambda message: remembered.append(message)
    ):
        pass
    assert remembered == [{"role": "assistant", "content": ""}]


@pytest.mark.asyncio
async def test_responses_stream_tool_call_without_arguments_stores_empty_object() -> None:
    async def chunks():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-empty",
                                "type": "function",
                                "function": {"name": "ping"},
                            }
                        ]
                    },
                    "index": 0,
                }
            ]
        }

    remembered: list[dict[str, object]] = []
    async for _ in aopenai_chunks_to_responses_sse(
        chunks(), model="test", on_complete=lambda message: remembered.append(message)
    ):
        pass
    assert remembered == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-empty",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        }
    ]


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses_stream_failure_does_not_remember_turn(tmp_path: Path) -> None:
    mock_models()
    error_stream = 'data: {"error":{"message":"upstream broke"}}\n\ndata: [DONE]\n\n'
    mock_completions(
        {
            FREE_A: [
                httpx.Response(
                    200,
                    content=error_stream.encode(),
                    headers={"content-type": "text/event-stream"},
                ),
                httpx.Response(200, json=completion_payload(FREE_A, "continued")),
            ],
            FREE_B: httpx.Response(
                200, content=error_stream.encode(), headers={"content-type": "text/event-stream"}
            ),
        }
    )
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        async with client.stream(
            "POST", "/v1/responses", json={"model": "freewhirr", "input": "start", "stream": True}
        ) as first:
            first_text = "".join([part async for part in first.aiter_text()])
        failed_id = json.loads(
            next(line[6:] for line in first_text.splitlines() if line.startswith("data: "))
        )["response"]["id"]
        followup = await client.post(
            "/v1/responses",
            json={
                "model": "freewhirr",
                "previous_response_id": failed_id,
                "input": "continue",
            },
        )
    assert "event: response.failed" in first_text
    assert followup.status_code == 200
    messages = json.loads(_last_completion_request().read())["messages"]
    assert all(message["role"] != "assistant" for message in messages)


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses_stream_text_continuation(tmp_path: Path) -> None:
    mock_models()
    first_stream = (
        'data: {"id":"s1","choices":[{"delta":{"content":"hello"},"index":0}]}'
        '\n\ndata: {"id":"s1","choices":[{"delta":{},"finish_reason":"stop"}]}'
        "\n\ndata: [DONE]\n\n"
    )
    done_stream = (
        'data: {"id":"s2","choices":[{"delta":{"content":"done"},"index":0}]}'
        '\n\ndata: {"id":"s2","choices":[{"delta":{},"finish_reason":"stop"}]}'
        "\n\ndata: [DONE]\n\n"
    )
    mock_completions(
        {
            FREE_A: [
                httpx.Response(
                    200,
                    content=first_stream.encode(),
                    headers={"content-type": "text/event-stream"},
                ),
                httpx.Response(
                    200, content=done_stream.encode(), headers={"content-type": "text/event-stream"}
                ),
            ]
        }
    )
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        async with client.stream(
            "POST", "/v1/responses", json={"model": "freewhirr", "input": "hi", "stream": True}
        ) as first:
            first_text = "".join([part async for part in first.aiter_text()])
        first_id = _completed_response(first_text)["id"]
        async with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "freewhirr",
                "previous_response_id": first_id,
                "input": "next",
                "stream": True,
            },
        ) as second:
            second_text = "".join([part async for part in second.aiter_text()])
    assert "event: response.completed" in second_text
    payload = json.loads(_last_completion_request().read())
    assert payload["messages"][-2] == {"role": "assistant", "content": "hello"}


@respx.mock
@pytest.mark.asyncio
async def test_proxy_responses_stream_ids_are_unique_and_continuable(tmp_path: Path) -> None:
    mock_models()
    streams = [
        'data: {"id":"s1","choices":[{"delta":{"content":"one"},"index":0}]}\n\n'
        'data: {"id":"s1","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n",
        'data: {"id":"s2","choices":[{"delta":{"content":"two"},"index":0}]}\n\n'
        'data: {"id":"s2","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n",
        'data: {"id":"s3","choices":[{"delta":{"content":"done"},"index":0}]}\n\n'
        'data: {"id":"s3","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n",
    ]
    mock_completions(
        {
            FREE_A: [
                httpx.Response(
                    200, content=stream.encode(), headers={"content-type": "text/event-stream"}
                )
                for stream in streams
            ]
        }
    )
    app = create_app(make_client(tmp_path))
    async with _http(app) as client:
        ids = []
        for prompt in ("first", "second"):
            async with client.stream(
                "POST",
                "/v1/responses",
                json={"model": "freewhirr", "input": prompt, "stream": True},
            ) as response:
                ids.append(
                    _completed_response("".join([part async for part in response.aiter_text()]))[
                        "id"
                    ]
                )
        assert ids[0] != ids[1]
        async with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "freewhirr",
                "previous_response_id": ids[0],
                "input": "continue",
                "stream": True,
            },
        ) as response:
            continuation_text = "".join([part async for part in response.aiter_text()])
    assert _completed_response(continuation_text)["previous_response_id"] == ids[0]
    payload = json.loads(_last_completion_request().read())
    assert payload["messages"][-2] == {"role": "assistant", "content": "one"}


def _completed_response(sse: str) -> dict[str, object]:
    for block in sse.split("\n\n"):
        if block.startswith("event: response.completed\n"):
            data = json.loads(block.split("data: ", 1)[1])
            return data["response"]
    raise AssertionError("response.completed event not found")


def _last_completion_request() -> httpx.Request:
    calls = [call for call in respx.calls if call.request.url.path.endswith("/chat/completions")]
    return calls[-1].request
