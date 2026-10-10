from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from freewhirr.errors import BudgetExceeded, FreeModelsExhausted, PaidFallbackExhausted
from freewhirr.quality import MinLengthCheck
from freewhirr.types import ExhaustionPolicy

from .conftest import (
    FREE_A,
    FREE_B,
    PAID,
    completion_payload,
    make_client,
    mock_completions,
    mock_models,
)


@respx.mock
def test_uses_first_free_model(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "ok"))})
    client = make_client(tmp_path)
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    assert result.model == FREE_A
    assert result.paid is False
    assert result.cost_usd == 0.0
    assert result.models_tried == [FREE_A]


@respx.mock
def test_failsover_on_429(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(429, json={"error": {"message": "Rate limited"}}),
            FREE_B: httpx.Response(200, json=completion_payload(FREE_B, "second")),
        }
    )
    client = make_client(tmp_path)
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.model == FREE_B
    assert result.content == "second"
    assert FREE_A in result.models_tried
    assert FREE_B in result.models_tried


@respx.mock
def test_failsover_on_context_length(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(
                400,
                json={"error": {"message": "This model's maximum context length was exceeded"}},
            ),
            FREE_B: httpx.Response(200, json=completion_payload(FREE_B, "fits")),
        }
    )
    client = make_client(tmp_path)
    result = client.complete(messages=[{"role": "user", "content": "huge"}])
    assert result.model == FREE_B


@respx.mock
def test_stop_policy_never_calls_paid(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(500, json={"error": {"message": "down"}}),
            FREE_B: httpx.Response(500, json={"error": {"message": "down"}}),
            PAID: httpx.Response(200, json=completion_payload(PAID, "paid", cost=0.01)),
        }
    )
    client = make_client(
        tmp_path,
        exhaustion_policy=ExhaustionPolicy.STOP,
        paid_models=[PAID],
    )
    with pytest.raises(FreeModelsExhausted, match="no money was spent"):
        client.complete(messages=[{"role": "user", "content": "hi"}])
    # The paid mock would succeed if called; ensure it was never requested.
    paid_calls = [
        call
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
        and json.loads(call.request.content)["model"] == PAID
    ]
    assert paid_calls == []


@respx.mock
def test_paid_policy_escalates_within_guardrails(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(429, json={"error": {"message": "rate"}}),
            FREE_B: httpx.Response(429, json={"error": {"message": "rate"}}),
            PAID: httpx.Response(200, json=completion_payload(PAID, "from paid", cost=0.002)),
        }
    )
    client = make_client(
        tmp_path,
        exhaustion_policy="paid",
        paid_models=[PAID],
        max_paid_per_request=1.0,
        max_paid_per_day=5.0,
    )
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.paid is True
    assert result.model == PAID
    assert result.cost_usd == 0.002


@respx.mock
def test_paid_blocked_by_daily_budget(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(500, json={"error": {"message": "x"}}),
            FREE_B: httpx.Response(500, json={"error": {"message": "x"}}),
            PAID: httpx.Response(200, json=completion_payload(PAID, "paid", cost=0.01)),
        }
    )
    client = make_client(
        tmp_path,
        exhaustion_policy="paid",
        paid_models=[PAID],
        max_paid_per_day=0.0000001,
        default_max_tokens=2048,
    )
    with pytest.raises(BudgetExceeded):
        client.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
def test_paid_only_high_complexity(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(500, json={"error": {"message": "x"}}),
            FREE_B: httpx.Response(500, json={"error": {"message": "x"}}),
            PAID: httpx.Response(200, json=completion_payload(PAID, "paid", cost=0.01)),
        }
    )
    client = make_client(
        tmp_path,
        exhaustion_policy="paid",
        paid_models=[PAID],
        paid_only_high_complexity=True,
    )
    with pytest.raises(BudgetExceeded, match="high-complexity"):
        client.complete(messages=[{"role": "user", "content": "hi"}])
    result = client.complete(
        messages=[{"role": "user", "content": "hard problem"}],
        complexity="high",
    )
    assert result.paid is True
    assert result.content == "paid"


@respx.mock
def test_quality_check_failsover(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "no")),
            FREE_B: httpx.Response(
                200, json=completion_payload(FREE_B, "a longer acceptable answer")
            ),
        }
    )
    client = make_client(tmp_path)
    result = client.complete(
        messages=[{"role": "user", "content": "hi"}],
        quality=MinLengthCheck(10),
    )
    assert result.model == FREE_B


@respx.mock
def test_privacy_sets_data_collection_deny(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A))})
    client = make_client(tmp_path, deny_data_collection=True)
    client.complete(messages=[{"role": "user", "content": "secret"}])
    posted = [
        json.loads(call.request.content)
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
    ]
    assert posted
    assert posted[0]["provider"] == {"data_collection": "deny"}
    assert posted[0]["model"] == FREE_A
    headers = [
        call.request.headers
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
    ]
    assert headers[0]["user-agent"].startswith("freewhirr/")


@respx.mock
def test_retries_then_failsover(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: [
                httpx.Response(429, json={"error": {"message": "rate"}}),
                httpx.Response(429, json={"error": {"message": "rate"}}),
            ],
            FREE_B: httpx.Response(200, json=completion_payload(FREE_B, "ok")),
        }
    )
    client = make_client(tmp_path, max_retries=1, retry_base_delay=0.0)
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.model == FREE_B
    a_calls = [
        json.loads(call.request.content)["model"]
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
    ]
    assert a_calls.count(FREE_A) == 2


@respx.mock
def test_stream_failsover_before_first_token(tmp_path: Path) -> None:
    mock_models()

    def _stream_ok() -> httpx.Response:
        body = (
            'data: {"id":"s1","choices":[{"delta":{"content":"hi"},"index":0}]}\n\n'
            'data: {"id":"s1","choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4,"cost":0}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    mock_completions(
        {
            FREE_A: httpx.Response(429, json={"error": {"message": "rate"}}),
            FREE_B: _stream_ok(),
        }
    )
    client = make_client(tmp_path)
    chunks = list(client.stream(messages=[{"role": "user", "content": "hi"}]))
    texts = [
        (choice.get("delta") or {}).get("content")
        for chunk in chunks
        for choice in chunk.get("choices") or []
    ]
    assert "hi" in texts
    assert chunks[0]["model"] == FREE_B


@respx.mock
@pytest.mark.asyncio
async def test_async_complete(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "async"))})
    client = make_client(tmp_path)
    result = await client.acomplete(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "async"
    await client.aclose()


@respx.mock
def test_openai_namespace(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "ns"))})
    client = make_client(tmp_path)
    result = client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert result.content == "ns"


@respx.mock
def test_all_paid_fail(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(500, json={"error": {"message": "x"}}),
            FREE_B: httpx.Response(500, json={"error": {"message": "x"}}),
            PAID: httpx.Response(500, json={"error": {"message": "x"}}),
        }
    )
    client = make_client(tmp_path, exhaustion_policy="paid", paid_models=[PAID])
    with pytest.raises(PaidFallbackExhausted):
        client.complete(messages=[{"role": "user", "content": "hi"}])


@respx.mock
def test_catalog_cached(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A))})
    client = make_client(tmp_path, include_discovered_free=True, free_models=[])
    client.complete(messages=[{"role": "user", "content": "a"}])
    client.complete(messages=[{"role": "user", "content": "b"}])
    model_calls = [call for call in respx.calls if call.request.url.path.endswith("/models")]
    assert len(model_calls) == 1
