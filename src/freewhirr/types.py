"""Shared enums and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExhaustionPolicy(str, Enum):
    """What to do after every free model has been tried."""

    STOP = "stop"
    PAID = "paid"


class Complexity(str, Enum):
    """Optional per-request tag used by the paid-escalation gate."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Outcome(str, Enum):
    SUCCESS_FREE = "success_free"
    SUCCESS_PAID = "success_paid"
    EXHAUSTED = "exhausted"
    ERROR = "error"


class FailReason(str, Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_LENGTH = "context_length"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    QUALITY = "quality"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelPricing:
    prompt: float = 0.0
    completion: float = 0.0
    request: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.prompt == 0.0 and self.completion == 0.0 and self.request == 0.0


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str = ""
    context_length: int | None = None
    pricing: ModelPricing = field(default_factory=ModelPricing)
    supports_tools: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_free(self) -> bool:
        return self.pricing.is_free or self.id.endswith(":free")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_response(cls, payload: dict[str, Any] | None) -> Usage:
        if not payload:
            return cls()
        usage = payload.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        cost = usage.get("cost")
        if cost is None:
            details = usage.get("cost_details") or {}
            cost = details.get("upstream_inference_cost")
        try:
            cost_usd = float(cost or 0.0)
        except (TypeError, ValueError):
            cost_usd = 0.0
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cost_usd=cost_usd,
        )


@dataclass
class Attempt:
    model: str
    reason: str
    status_code: int | None = None
    message: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reason": self.reason,
            "status_code": self.status_code,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class ChatResult:
    """A completed chat call plus the routing metadata that produced it."""

    raw: dict[str, Any]
    model: str
    models_tried: list[str]
    attempts: list[Attempt]
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    usage: Usage
    paid: bool
    request_id: str

    @property
    def content(self) -> str:
        return extract_message_content(self.raw)

    @property
    def id(self) -> str | None:
        value = self.raw.get("id")
        return str(value) if value is not None else None

    @property
    def choices(self) -> list[Any]:
        choices = self.raw.get("choices")
        return choices if isinstance(choices, list) else []

    def to_openai(self) -> dict[str, Any]:
        """Return the upstream payload, ensuring the winning model is set."""
        payload = dict(self.raw)
        payload["model"] = self.model
        return payload


def extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    text = first.get("text")
    return text if isinstance(text, str) else ""
