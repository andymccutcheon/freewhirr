from __future__ import annotations

from freewhirr.retry import backoff_delays, classify_failure
from freewhirr.types import FailReason


def test_classify_rate_limit() -> None:
    reason, retryable = classify_failure(status_code=429, message="Too many requests")
    assert reason is FailReason.RATE_LIMIT
    assert retryable


def test_classify_context_length() -> None:
    reason, retryable = classify_failure(
        status_code=400,
        body={"error": {"message": "This model's maximum context length is 8192 tokens"}},
    )
    assert reason is FailReason.CONTEXT_LENGTH
    assert not retryable


def test_classify_timeout() -> None:
    reason, retryable = classify_failure(timeout=True)
    assert reason is FailReason.TIMEOUT
    assert retryable


def test_classify_server_error() -> None:
    reason, retryable = classify_failure(status_code=503, message="unavailable")
    assert reason is FailReason.SERVER_ERROR
    assert retryable


def test_backoff_grows_and_is_positive() -> None:
    delays = list(backoff_delays(3, base=0.5, cap=8.0))
    assert len(delays) == 3
    assert delays[0] >= 0.5
    assert delays[1] >= delays[0]
    assert delays[2] >= delays[1]
    assert all(delay <= 8.0 * 1.5 for delay in delays)
