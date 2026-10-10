"""Optional post-response quality checks that trigger failover."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from freeloader.types import extract_message_content


class QualityCheck(Protocol):
    """Return True when the response is acceptable."""

    def __call__(self, payload: dict[str, Any]) -> bool: ...


class MinLengthCheck:
    """Reject answers shorter than ``minimum`` characters."""

    def __init__(self, minimum: int) -> None:
        self.minimum = minimum

    def __call__(self, payload: dict[str, Any]) -> bool:
        return len(extract_message_content(payload)) >= self.minimum


class JsonSchemaCheck:
    """Parse the assistant message as JSON and validate it against a schema."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def __call__(self, payload: dict[str, Any]) -> bool:
        import json

        import jsonschema

        content = extract_message_content(payload).strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return False
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.ValidationError:
            return False
        return True


class CallableCheck:
    """Wrap a user callable that receives the raw OpenAI-style payload."""

    def __init__(self, fn: Callable[[dict[str, Any]], bool]) -> None:
        self.fn = fn

    def __call__(self, payload: dict[str, Any]) -> bool:
        return bool(self.fn(payload))


def coerce_quality_check(
    check: QualityCheck | Callable[[dict[str, Any]], bool] | None,
) -> QualityCheck | None:
    if check is None:
        return None
    if isinstance(check, (MinLengthCheck, JsonSchemaCheck, CallableCheck)):
        return check
    return CallableCheck(check)
