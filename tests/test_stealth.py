from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from freewhirr.cli import main
from freewhirr.models import parse_model
from freewhirr.stealth import diff_and_score, score_stealth
from freewhirr.types import ExhaustionPolicy

from .conftest import (
    FREE_A,
    FREE_B,
    OPENROUTER,
    completion_payload,
    make_client,
    mock_completions,
    mock_models,
)

HORIZON = "openrouter/horizon-beta"
LOGGER = "vendor/logger:free"

STEALTH_CATALOG = {
    "data": [
        {
            "id": FREE_A,
            "name": "Llama 3.2 3B",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "supported_parameters": ["temperature"],
            "description": "A normal free model from a known lab.",
        },
        {
            "id": FREE_B,
            "name": "Gemma 2 9B",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "supported_parameters": ["tools", "tool_choice"],
            "description": "Another ordinary free model.",
        },
        {
            "id": HORIZON,
            "name": "Horizon Beta",
            "canonical_slug": "openrouter/horizon-beta",
            "context_length": 256000,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "supported_parameters": ["tools", "tool_choice"],
            "description": (
                "This is a cloaked model provided to the community to gather feedback. "
                "It's free to use during this testing period, and prompts and completions "
                "are logged by the model creator for feedback and training."
            ),
        },
        {
            "id": LOGGER,
            "name": "Logger",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "description": "Prompts are logged and used for training.",
        },
        {
            "id": "openrouter/free",
            "name": "Free Models Router",
            "context_length": 200000,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
            "description": "Router that picks a free model at random.",
        },
    ]
}


def _mock_stealth_catalog() -> None:
    respx.get(f"{OPENROUTER}/models").mock(return_value=httpx.Response(200, json=STEALTH_CATALOG))


def test_horizon_shape_scores_as_stealth() -> None:
    info = parse_model(STEALTH_CATALOG["data"][2])
    report = score_stealth(info, is_new=True)
    assert report.is_stealth
    assert report.logs_or_trains
    assert report.score >= 6
    assert any("cloaked" in reason for reason in report.reasons)
    assert any("openrouter/" in reason for reason in report.reasons)


def test_known_router_is_not_stealth() -> None:
    info = parse_model(STEALTH_CATALOG["data"][4])
    report = score_stealth(info, is_new=True)
    assert report.score == 0
    assert not report.is_stealth


def test_ordinary_free_model_is_not_stealth() -> None:
    info = parse_model(STEALTH_CATALOG["data"][0])
    report = score_stealth(info, is_new=True)
    assert not report.is_stealth
    assert report.score <= 1


@respx.mock
def test_catalog_diff_flags_new_models(tmp_path: Path) -> None:
    _mock_stealth_catalog()
    client = make_client(tmp_path, include_discovered_free=True)
    new_ids, reports = diff_and_score(client)
    assert HORIZON in new_ids
    assert FREE_A in new_ids
    again, second = diff_and_score(client)
    assert again == []
    assert any(item.is_stealth and item.model_id == HORIZON for item in second)


@respx.mock
def test_promote_stealth_fronts_the_queue(tmp_path: Path) -> None:
    _mock_stealth_catalog()
    mock_completions(
        {HORIZON: httpx.Response(200, json=completion_payload(HORIZON, "from stealth"))}
    )
    client = make_client(
        tmp_path,
        include_discovered_free=True,
        promote_stealth=True,
        free_models=[FREE_A, FREE_B],
    )
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.model == HORIZON


@respx.mock
def test_deny_data_collection_excludes_stealth_and_loggers(tmp_path: Path) -> None:
    _mock_stealth_catalog()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "safe"))})
    client = make_client(
        tmp_path,
        include_discovered_free=True,
        promote_stealth=True,
        deny_data_collection=True,
        free_models=[HORIZON, LOGGER, FREE_A],
    )
    assert HORIZON not in client.list_free_models()
    assert LOGGER not in client.list_free_models()
    assert FREE_A in client.list_free_models()
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.model == FREE_A
    _new, reports = diff_and_score(client)
    horizon = next(item for item in reports if item.model_id == HORIZON)
    assert horizon.is_stealth
    assert horizon.excluded


@respx.mock
def test_watch_once_and_webhook(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_stealth_catalog()
    webhook = "https://alerts.example.test/hook"
    respx.post(webhook).mock(return_value=httpx.Response(204))
    config = tmp_path / "freewhirr.yaml"
    config.write_text(
        "\n".join(
            [
                "api_key: test-key",
                f"state_path: {tmp_path / 'state.sqlite'}",
                "include_discovered_free: true",
                "alert_webhook: https://alerts.example.test/hook",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert main(["--config", str(config), "watch", "--once"]) == 0
    out = capsys.readouterr().out
    assert "stealth=" in out or HORIZON in out
    posted = [call for call in respx.calls if str(call.request.url) == webhook]
    assert posted
    assert main(["--config", str(config), "models", "--stealth"]) == 0
    stealth_out = capsys.readouterr().out
    assert HORIZON in stealth_out
    assert main(["--config", str(config), "models", "--new"]) == 0


@respx.mock
def test_stop_still_skips_paid_when_stealth_promoted(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "ok"))})
    client = make_client(
        tmp_path,
        exhaustion_policy=ExhaustionPolicy.STOP,
        promote_stealth=True,
    )
    result = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert result.model == FREE_A
    assert result.paid is False
