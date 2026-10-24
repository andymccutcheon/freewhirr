"""Streaming empty-reply guard and simulated 429 failover (no real provider calls)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from freewhirr.errors import FreeModelsExhausted

from .conftest import FREE_A, FREE_B, PAID, make_client, mock_completions, mock_models

MSG = [{"role": "user", "content": "hi"}]


def _sse(*deltas: dict, finish: str = "stop") -> httpx.Response:
    lines = [
        "data: " + json.dumps({"id": "s", "choices": [{"index": 0, "delta": d}]}) for d in deltas
    ]
    lines.append(
        "data: "
        + json.dumps(
            {
                "id": "s",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4, "cost": 0},
            }
        )
    )
    lines.append("data: [DONE]")
    return httpx.Response(
        200,
        content=("\n\n".join(lines) + "\n\n").encode(),
        headers={"content-type": "text/event-stream"},
    )


REASONING_ONLY = (
    {"role": "assistant", "content": "", "reasoning": "thinking"},
    {"content": "", "reasoning": " more"},
)


def _texts(chunks: list[dict]) -> str:
    return "".join(
        (c.get("delta") or {}).get("content") or ""
        for chunk in chunks
        for c in chunk.get("choices") or []
    )


@respx.mock
def test_stream_empty_reply_fails_over(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: _sse(*REASONING_ONLY), FREE_B: _sse({"content": "ok"})})
    client = make_client(tmp_path, free_models=[FREE_A, FREE_B], include_discovered_free=False)
    chunks = list(client.stream(messages=MSG))
    assert _texts(chunks) == "ok"
    assert all(chunk["model"] == FREE_B for chunk in chunks)


@respx.mock
def test_stream_tool_call_only_is_valid(tmp_path: Path) -> None:
    mock_models()
    call = {
        "tool_calls": [
            {
                "index": 0,
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
    }
    mock_completions({FREE_A: _sse(*REASONING_ONLY, call, finish="tool_calls")})
    client = make_client(tmp_path, free_models=[FREE_A], include_discovered_free=False)
    chunks = list(client.stream(messages=MSG))
    assert chunks[0]["model"] == FREE_A
    assert any(
        (c.get("delta") or {}).get("tool_calls") for chunk in chunks for c in chunk["choices"]
    )
    # held reasoning chunks are released in order, ahead of the tool call
    assert (chunks[0]["choices"][0]["delta"]).get("reasoning") == "thinking"


@respx.mock
def test_stream_refusal_is_valid(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: _sse({"content": "", "refusal": "I can't help with that"})})
    client = make_client(tmp_path, free_models=[FREE_A], include_discovered_free=False)
    assert chunks_model(list(client.stream(messages=MSG))) == FREE_A


def chunks_model(chunks: list[dict]) -> str:
    return chunks[0]["model"]


@respx.mock
def test_stream_empty_allowed_when_guard_disabled(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: _sse(*REASONING_ONLY)})
    client = make_client(
        tmp_path,
        free_models=[FREE_A],
        include_discovered_free=False,
        reject_empty_responses=False,
    )
    assert chunks_model(list(client.stream(messages=MSG))) == FREE_A


@respx.mock
def test_stream_all_empty_raises_exhausted_without_paid(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {FREE_A: _sse(*REASONING_ONLY), FREE_B: _sse(*REASONING_ONLY), PAID: _sse({"content": "x"})}
    )
    client = make_client(
        tmp_path,
        free_models=[FREE_A, FREE_B],
        include_discovered_free=False,
        exhaustion_policy="stop",
        paid_models=[PAID],
    )
    with pytest.raises(FreeModelsExhausted):
        list(client.stream(messages=MSG))
    called = [
        json.loads(c.request.content)["model"]
        for c in respx.calls
        if c.request.url.path.endswith("/chat/completions")
    ]
    assert PAID not in called


@respx.mock
async def test_async_stream_empty_reply_fails_over(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: _sse(*REASONING_ONLY), FREE_B: _sse({"content": "ok"})})
    client = make_client(tmp_path, free_models=[FREE_A, FREE_B], include_discovered_free=False)
    chunks = [chunk async for chunk in client.astream(messages=MSG)]
    assert _texts(chunks) == "ok"
    assert all(chunk["model"] == FREE_B for chunk in chunks)


@respx.mock
async def test_async_stream_tool_call_only_is_valid(tmp_path: Path) -> None:
    mock_models()
    call = {
        "tool_calls": [
            {
                "index": 0,
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ]
    }
    mock_completions({FREE_A: _sse(*REASONING_ONLY, call, finish="tool_calls")})
    client = make_client(tmp_path, free_models=[FREE_A], include_discovered_free=False)
    chunks = [chunk async for chunk in client.astream(messages=MSG)]
    assert chunks[0]["model"] == FREE_A


@respx.mock
def test_simulated_429_all_free_never_calls_paid(tmp_path: Path) -> None:
    mock_models()
    rate = lambda: httpx.Response(429, json={"error": {"message": "rate"}})  # noqa: E731
    mock_completions(
        {FREE_A: [rate()] * 4, FREE_B: [rate()] * 4, PAID: httpx.Response(200, json={})}
    )
    client = make_client(
        tmp_path,
        free_models=[FREE_A, FREE_B],
        include_discovered_free=False,
        exhaustion_policy="stop",
        paid_models=[PAID],
        max_retries=1,
        retry_base_delay=0.0,
    )
    with pytest.raises(FreeModelsExhausted) as exc:
        client.complete(messages=MSG)
    assert {a["reason"] for a in exc.value.attempts} == {"rate_limit"}
    called = [
        json.loads(c.request.content)["model"]
        for c in respx.calls
        if c.request.url.path.endswith("/chat/completions")
    ]
    assert PAID not in called and FREE_A in called and FREE_B in called


@respx.mock
def test_simulated_429_then_next_free_model_serves(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: [httpx.Response(429, json={"error": {"message": "rate"}})] * 4,
            FREE_B: httpx.Response(
                200,
                json={
                    "id": "g",
                    "model": FREE_B,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            ),
        }
    )
    client = make_client(
        tmp_path,
        free_models=[FREE_A, FREE_B],
        include_discovered_free=False,
        max_retries=1,
        retry_base_delay=0.0,
    )
    result = client.complete(messages=MSG)
    assert result.model == FREE_B and result.content == "ok"
    assert result.attempts[0].reason == "rate_limit"
