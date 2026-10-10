"""Jittered exponential backoff and error classification."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from freewhirr.types import FailReason

CONTEXT_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "max_tokens",
    "token limit",
    "maximum number of tokens",
    "prompt is too long",
    "exceeds the model's",
    "exceeds model",
)


def backoff_delays(max_retries: int, base: float, cap: float) -> Iterator[float]:
    """Yield sleep durations for *additional* attempts after the first try."""
    for attempt in range(max_retries):
        delay = min(cap, base * (2**attempt))
        jitter = random.uniform(0.0, delay * 0.5) if delay else 0.0
        yield delay + jitter


def _blob(status_code: int | None, body: Any, message: str = "") -> str:
    parts = [message, str(status_code or "")]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            parts.append(str(error.get("message") or ""))
            parts.append(str(error.get("code") or ""))
            meta = error.get("metadata")
            if meta is not None:
                parts.append(str(meta))
        parts.append(str(body))
    elif body is not None:
        parts.append(str(body))
    return " ".join(parts).lower()


def classify_failure(
    *,
    status_code: int | None = None,
    body: Any = None,
    message: str = "",
    timeout: bool = False,
) -> tuple[FailReason, bool]:
    """Return (reason, retry_same_model)."""
    if timeout:
        return FailReason.TIMEOUT, True
    text = _blob(status_code, body, message)
    if any(marker in text for marker in CONTEXT_MARKERS):
        return FailReason.CONTEXT_LENGTH, False
    if status_code == 429 or "rate limit" in text or "too many requests" in text:
        return FailReason.RATE_LIMIT, True
    if status_code == 408:
        return FailReason.TIMEOUT, True
    if status_code is not None and status_code >= 500:
        return FailReason.SERVER_ERROR, True
    if status_code is not None and 400 <= status_code < 500:
        return FailReason.CLIENT_ERROR, False
    if "timeout" in text:
        return FailReason.TIMEOUT, True
    return FailReason.UNKNOWN, True
