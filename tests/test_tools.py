from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from freewhirr.errors import FreeModelsExhausted

from .conftest import (
    FREE_A,
    FREE_B,
    LOOKUP_TOOL,
    PAID,
    completion_payload,
    make_client,
    mock_completions,
    mock_models,
    tool_completion_payload,
)


def _tool_calls() -> list[dict]:
    return [{"role": "user", "content": "lookup x"}]


@respx.mock
def test_skips_models_without_tool_metadata(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "should not be used")),
            FREE_B: httpx.Response(200, json=tool_completion_payload(FREE_B)),
        }
    )
    client = make_client(tmp_path)
    result = client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    assert result.model == FREE_B
    posted = [
        json.loads(call.request.content)["model"]
        for call in respx.calls
        if call.request.url.path.endswith("/chat/completions")
    ]
    assert FREE_A not in posted
    assert FREE_B in posted
    skipped = [attempt for attempt in result.attempts if attempt.reason == "skipped"]
    assert skipped and skipped[0].model == FREE_A


@respx.mock
def test_require_tool_support_can_be_disabled(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "no tools"))})
    client = make_client(tmp_path, require_tool_support=False, validate_tool_calls=False)
    result = client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    assert result.model == FREE_A


@respx.mock
def test_unknown_catalog_model_is_still_tried(tmp_path: Path) -> None:
    mock_models()
    mystery = "vendor/mystery-tool-model"
    mock_completions({mystery: httpx.Response(200, json=tool_completion_payload(mystery))})
    client = make_client(tmp_path, free_models=[mystery], include_discovered_free=False)
    result = client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    assert result.model == mystery


@respx.mock
def test_malformed_tool_json_failsover(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_B: httpx.Response(
                200, json=tool_completion_payload(FREE_B, arguments="{not-json")
            ),
            PAID: httpx.Response(200, json=tool_completion_payload(PAID)),
        }
    )
    client = make_client(
        tmp_path,
        free_models=[FREE_B],
        include_discovered_free=False,
        exhaustion_policy="paid",
        paid_models=[PAID],
    )
    result = client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    assert result.model == PAID
    assert any(attempt.reason == "quality" for attempt in result.attempts)


@respx.mock
def test_unknown_tool_name_failsover(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {
            FREE_B: httpx.Response(
                200, json=tool_completion_payload(FREE_B, name="not_a_real_tool")
            ),
            PAID: httpx.Response(200, json=tool_completion_payload(PAID)),
        }
    )
    client = make_client(
        tmp_path,
        free_models=[FREE_B],
        include_discovered_free=False,
        exhaustion_policy="paid",
        paid_models=[PAID],
    )
    result = client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    assert result.model == PAID


@respx.mock
def test_validate_tool_calls_can_be_disabled(tmp_path: Path) -> None:
    mock_models()
    mock_completions(
        {FREE_B: httpx.Response(200, json=tool_completion_payload(FREE_B, arguments="{nope"))}
    )
    client = make_client(tmp_path, validate_tool_calls=False)
    result = client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    assert result.model == FREE_B


@respx.mock
def test_all_models_skipped_when_none_support_tools(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "unused"))})
    client = make_client(tmp_path, free_models=[FREE_A], include_discovered_free=False)
    with pytest.raises(FreeModelsExhausted):
        client.complete(messages=_tool_calls(), tools=[LOOKUP_TOOL])
    posted = [call for call in respx.calls if call.request.url.path.endswith("/chat/completions")]
    assert posted == []
