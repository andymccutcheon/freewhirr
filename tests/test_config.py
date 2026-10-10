from __future__ import annotations

from pathlib import Path

import pytest

from freeloader.config import FreeloaderConfig, discover_config_path, load_config
from freeloader.errors import ConfigError
from freeloader.types import ExhaustionPolicy


def test_yaml_config_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "freeloader.yaml"
    path.write_text(
        """
exhaustion_policy: paid
paid_models:
  - openai/gpt-4o-mini
max_paid_per_day: 1.5
deny_data_collection: true
api_key: from-file
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    monkeypatch.setenv("FREELOADER_EXHAUSTION_POLICY", "stop")
    config = load_config(path)
    assert config.api_key == "from-env"
    assert config.exhaustion_policy is ExhaustionPolicy.STOP
    assert config.paid_models == ["openai/gpt-4o-mini"]
    assert config.max_paid_per_day == 1.5
    assert config.deny_data_collection is True


def test_toml_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "freeloader.toml"
    path.write_text(
        """
exhaustion_policy = "stop"
free_models = ["a/free", "b/free"]
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = load_config(path)
    assert config.free_models == ["a/free", "b/free"]
    assert config.exhaustion_policy is ExhaustionPolicy.STOP


def test_discover_cwd_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "freeloader.yaml").write_text("timeout: 12\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FREELOADER_CONFIG", raising=False)
    found = discover_config_path()
    assert found == tmp_path / "freeloader.yaml"


def test_comma_separated_env_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREELOADER_FREE_MODELS", "one, two ,three")
    monkeypatch.setenv("FREELOADER_PAID_MODELS", "paid/one")
    config = load_config()
    assert config.free_models == ["one", "two", "three"]
    assert config.paid_models == ["paid/one"]


def test_paid_policy_requires_models() -> None:
    config = FreeloaderConfig(exhaustion_policy="paid", paid_models=[])
    with pytest.raises(ConfigError, match="paid_models"):
        config.validate_policy()


def test_invalid_share() -> None:
    with pytest.raises(Exception):
        FreeloaderConfig(max_paid_share=1.5)
