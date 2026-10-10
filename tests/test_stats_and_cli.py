from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from freeloader.cli import main
from freeloader.stats import collect_stats

from .conftest import FREE_A, completion_payload, make_client, mock_completions, mock_models


@respx.mock
def test_stats_report(tmp_path: Path) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "ok"))})
    client = make_client(tmp_path)
    client.complete(messages=[{"role": "user", "content": "hi"}])
    report = collect_stats(client.store)
    assert report.requests == 1
    assert report.free_successes == 1
    assert report.paid_successes == 0
    assert report.spent_usd == 0.0
    assert report.saved_usd > 0
    text = report.format()
    assert "Estimated saved" in text
    assert "Free successes" in text


@respx.mock
def test_cli_chat_and_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    mock_models()
    mock_completions({FREE_A: httpx.Response(200, json=completion_payload(FREE_A, "cli-hi"))})
    config = tmp_path / "freeloader.yaml"
    config.write_text(
        "\n".join(
            [
                "api_key: test-key",
                f"state_path: {tmp_path / 'state.sqlite'}",
                "free_models:",
                f"  - {FREE_A}",
                "include_discovered_free: false",
                "max_retries: 0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert main(["--config", str(config), "chat", "hello"]) == 0
    out = capsys.readouterr().out
    assert "cli-hi" in out
    assert main(["--config", str(config), "stats"]) == 0
    stats_out = capsys.readouterr().out
    assert "Free successes" in stats_out


def test_cli_config_redacts_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "freeloader.yaml"
    config.write_text("api_key: super-secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--config", str(config), "config"]) == 0
    out = capsys.readouterr().out
    assert "super-secret" not in out
    assert "***" in out
