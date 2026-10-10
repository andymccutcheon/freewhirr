"""Configuration from defaults, a YAML/TOML file, environment variables, and kwargs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from freewhirr.errors import ConfigError
from freewhirr.types import ExhaustionPolicy

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_APP_URL = "https://github.com/andymccutcheon/freewhirr"
DEFAULT_APP_TITLE = "freewhirr"
CONFIG_FILENAMES = (
    "freewhirr.yaml",
    "freewhirr.yml",
    "freewhirr.toml",
)


class FreewhirrConfig(BaseModel):
    """All knobs. Nothing here locks you into a vendor besides OpenRouter itself."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    exhaustion_policy: ExhaustionPolicy = ExhaustionPolicy.STOP
    free_models: list[str] = Field(default_factory=list)
    paid_models: list[str] = Field(default_factory=list)
    include_discovered_free: bool = True
    max_paid_per_request: float | None = None
    max_paid_per_day: float | None = None
    max_paid_per_month: float | None = None
    max_paid_share: float | None = None
    paid_only_high_complexity: bool = False
    deny_data_collection: bool = False
    model_cache_ttl: float = 3600.0
    timeout: float = 60.0
    max_retries: int = 2
    retry_base_delay: float = 0.5
    retry_cap: float = 8.0
    state_path: Path = Field(default_factory=lambda: Path.home() / ".freewhirr" / "state.sqlite")
    state_backend: str = "sqlite"
    baseline_paid_model: str = "openai/gpt-4o-mini"
    honor_requested_model: bool = False
    fetch_generation_cost: bool = False
    require_tool_support: bool = True
    validate_tool_calls: bool = True
    app_url: str = DEFAULT_APP_URL
    app_title: str = DEFAULT_APP_TITLE
    log_level: str = "INFO"
    default_max_tokens: int = 1024

    @field_validator("exhaustion_policy", mode="before")
    @classmethod
    def _coerce_policy(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("free_models", "paid_models", mode="before")
    @classmethod
    def _split_models(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("state_path", mode="before")
    @classmethod
    def _expand_path(cls, value: Any) -> Any:
        if value is None or value == "":
            return Path.home() / ".freewhirr" / "state.sqlite"
        return Path(value).expanduser()

    @field_validator("state_backend")
    @classmethod
    def _backend(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"sqlite", "json"}:
            raise ValueError("state_backend must be 'sqlite' or 'json'")
        return normalized

    @field_validator("max_paid_share")
    @classmethod
    def _share(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError("max_paid_share must be between 0 and 1 (a fraction, not a percent)")
        return value

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigError(
                "Missing OpenRouter API key. Set OPENROUTER_API_KEY or api_key in your config file."
            )
        return self.api_key

    def validate_policy(self) -> None:
        if self.exhaustion_policy is ExhaustionPolicy.PAID and not self.paid_models:
            raise ConfigError(
                "exhaustion_policy is 'paid' but paid_models is empty. "
                "List the paid models you are willing to escalate to."
            )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if stripped == "" or stripped.lower() in {"none", "null"}:
        return None
    return float(stripped)


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        loaded = yaml.safe_load(text) or {}
    elif suffix == ".toml":
        loaded = tomllib.loads(text)
    else:
        raise ConfigError(f"Unsupported config format: {path.suffix} (use .yaml, .yml, or .toml)")
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file must contain a mapping at the top level: {path}")
    return loaded


def discover_config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.environ.get("FREEWHIRR_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    cwd = Path.cwd()
    for name in CONFIG_FILENAMES:
        candidate = cwd / name
        if candidate.is_file():
            return candidate
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    for name in CONFIG_FILENAMES:
        candidate = config_home / "freewhirr" / name
        if candidate.is_file():
            return candidate
    return None


def env_overrides() -> dict[str, Any]:
    """Map documented environment variables onto config fields."""
    mapping: dict[str, Any] = {}
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        mapping["api_key"] = key
    if os.environ.get("OPENROUTER_BASE_URL"):
        mapping["base_url"] = os.environ["OPENROUTER_BASE_URL"].rstrip("/")
    policy = os.environ.get("FREEWHIRR_EXHAUSTION_POLICY") or os.environ.get("FREEWHIRR_POLICY")
    if policy:
        mapping["exhaustion_policy"] = policy
    if os.environ.get("FREEWHIRR_FREE_MODELS"):
        mapping["free_models"] = os.environ["FREEWHIRR_FREE_MODELS"]
    if os.environ.get("FREEWHIRR_PAID_MODELS"):
        mapping["paid_models"] = os.environ["FREEWHIRR_PAID_MODELS"]
    if os.environ.get("FREEWHIRR_INCLUDE_DISCOVERED"):
        mapping["include_discovered_free"] = _parse_bool(os.environ["FREEWHIRR_INCLUDE_DISCOVERED"])
    if os.environ.get("FREEWHIRR_MAX_PAID_PER_REQUEST"):
        mapping["max_paid_per_request"] = _parse_optional_float(
            os.environ["FREEWHIRR_MAX_PAID_PER_REQUEST"]
        )
    if os.environ.get("FREEWHIRR_MAX_PAID_PER_DAY"):
        mapping["max_paid_per_day"] = _parse_optional_float(
            os.environ["FREEWHIRR_MAX_PAID_PER_DAY"]
        )
    if os.environ.get("FREEWHIRR_MAX_PAID_PER_MONTH"):
        mapping["max_paid_per_month"] = _parse_optional_float(
            os.environ["FREEWHIRR_MAX_PAID_PER_MONTH"]
        )
    if os.environ.get("FREEWHIRR_MAX_PAID_SHARE"):
        mapping["max_paid_share"] = _parse_optional_float(os.environ["FREEWHIRR_MAX_PAID_SHARE"])
    if os.environ.get("FREEWHIRR_PAID_ONLY_HIGH_COMPLEXITY"):
        mapping["paid_only_high_complexity"] = _parse_bool(
            os.environ["FREEWHIRR_PAID_ONLY_HIGH_COMPLEXITY"]
        )
    if os.environ.get("FREEWHIRR_DENY_DATA_COLLECTION"):
        mapping["deny_data_collection"] = _parse_bool(os.environ["FREEWHIRR_DENY_DATA_COLLECTION"])
    if os.environ.get("FREEWHIRR_MODEL_CACHE_TTL"):
        mapping["model_cache_ttl"] = float(os.environ["FREEWHIRR_MODEL_CACHE_TTL"])
    if os.environ.get("FREEWHIRR_TIMEOUT"):
        mapping["timeout"] = float(os.environ["FREEWHIRR_TIMEOUT"])
    if os.environ.get("FREEWHIRR_MAX_RETRIES"):
        mapping["max_retries"] = int(os.environ["FREEWHIRR_MAX_RETRIES"])
    if os.environ.get("FREEWHIRR_STATE_PATH"):
        mapping["state_path"] = os.environ["FREEWHIRR_STATE_PATH"]
    if os.environ.get("FREEWHIRR_STATE_BACKEND"):
        mapping["state_backend"] = os.environ["FREEWHIRR_STATE_BACKEND"]
    if os.environ.get("FREEWHIRR_BASELINE_MODEL"):
        mapping["baseline_paid_model"] = os.environ["FREEWHIRR_BASELINE_MODEL"]
    if os.environ.get("FREEWHIRR_HONOR_REQUESTED_MODEL"):
        mapping["honor_requested_model"] = _parse_bool(
            os.environ["FREEWHIRR_HONOR_REQUESTED_MODEL"]
        )
    if os.environ.get("FREEWHIRR_REQUIRE_TOOL_SUPPORT"):
        mapping["require_tool_support"] = _parse_bool(
            os.environ["FREEWHIRR_REQUIRE_TOOL_SUPPORT"]
        )
    if os.environ.get("FREEWHIRR_VALIDATE_TOOL_CALLS"):
        mapping["validate_tool_calls"] = _parse_bool(os.environ["FREEWHIRR_VALIDATE_TOOL_CALLS"])
    if os.environ.get("FREEWHIRR_APP_URL"):
        mapping["app_url"] = os.environ["FREEWHIRR_APP_URL"]
    if os.environ.get("FREEWHIRR_APP_TITLE"):
        mapping["app_title"] = os.environ["FREEWHIRR_APP_TITLE"]
    if os.environ.get("FREEWHIRR_LOG_LEVEL"):
        mapping["log_level"] = os.environ["FREEWHIRR_LOG_LEVEL"]
    return mapping


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> FreewhirrConfig:
    """Merge defaults < file < env < explicit overrides. Env always wins over the file."""
    data: dict[str, Any] = {}
    found = discover_config_path(path)
    if found is not None:
        data.update(_read_config_file(found))
    data.update(env_overrides())
    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    try:
        config = FreewhirrConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(str(exc)) from exc
    return config
