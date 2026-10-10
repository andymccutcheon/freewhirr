"""Exception hierarchy for freeloader."""

from __future__ import annotations

from typing import Any


class FreeloaderError(Exception):
    """Base error for every freeloader failure."""


class ConfigError(FreeloaderError):
    """Invalid configuration or a missing required setting."""


class OpenRouterError(FreeloaderError):
    """An upstream OpenRouter / HTTP failure that is not a failover signal."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: Any = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.model = model


class FailoverError(FreeloaderError):
    """A single-model attempt failed in a way that should try the next model."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        model: str,
        status_code: int | None = None,
        body: Any = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.model = model
        self.status_code = status_code
        self.body = body
        self.retryable = retryable


class QualityFailed(FailoverError):
    """The model responded, but a quality check rejected the output."""

    def __init__(self, message: str, *, model: str, body: Any = None) -> None:
        super().__init__(
            message,
            reason="quality",
            model=model,
            body=body,
            retryable=False,
        )


class FreeModelsExhausted(FreeloaderError):
    """Every free model failed or was skipped, and paid escalation is not allowed.

    This is the `stop` policy outcome, and also the `paid` policy outcome once a
    budget, share cap, or complexity gate blocks escalation.
    """

    def __init__(
        self,
        message: str = "All free models are exhausted and paid fallback is not allowed.",
        *,
        attempts: list[dict[str, Any]] | None = None,
        reason: str = "stop",
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []
        self.reason = reason


class BudgetExceeded(FreeModelsExhausted):
    """Paid fallback is configured, but a spend or share guardrail was hit."""

    def __init__(
        self,
        message: str = "Paid budget or share cap reached; refusing to spend more.",
        *,
        attempts: list[dict[str, Any]] | None = None,
        reason: str = "budget",
    ) -> None:
        super().__init__(message, attempts=attempts, reason=reason)


class PaidFallbackExhausted(FreeloaderError):
    """Free models failed and every eligible paid model also failed."""

    def __init__(
        self,
        message: str = "Free and paid model lists are both exhausted.",
        *,
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []
