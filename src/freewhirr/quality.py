"""Optional post-response quality checks that trigger failover."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from freewhirr.types import extract_message_content


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


class ToolCallCheck:
    """Reject malformed tool calls: unknown names or arguments that are not JSON."""

    def __init__(self, allowed_names: set[str] | list[str] | None = None) -> None:
        self.allowed_names = {str(name) for name in (allowed_names or []) if name}

    def __call__(self, payload: dict[str, Any]) -> bool:
        choices = payload.get("choices") or []
        if not choices:
            return True
        first = choices[0] or {}
        message = first.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        finish = first.get("finish_reason")
        if finish == "tool_calls" and not tool_calls:
            return False
        if not tool_calls:
            return True
        for call in tool_calls:
            if not isinstance(call, dict):
                return False
            function = call.get("function") or {}
            if not isinstance(function, dict):
                return False
            name = str(function.get("name") or "")
            if not name:
                return False
            if self.allowed_names and name not in self.allowed_names:
                return False
            raw_args = function.get("arguments")
            if raw_args in (None, ""):
                continue
            if isinstance(raw_args, dict):
                continue
            if not isinstance(raw_args, str):
                return False
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                return False
            if not isinstance(parsed, (dict, list)):
                return False
        return True


class AllChecks:
    """Require every nested check to accept the payload."""

    def __init__(self, checks: list[QualityCheck]) -> None:
        self.checks = list(checks)

    def __call__(self, payload: dict[str, Any]) -> bool:
        return all(check(payload) for check in self.checks)


_BUILTIN_CHECKS = (
    MinLengthCheck,
    JsonSchemaCheck,
    CallableCheck,
    ToolCallCheck,
    AllChecks,
)


def request_uses_tools(*sources: dict[str, Any] | None) -> bool:
    for source in sources:
        if not source:
            continue
        tools = source.get("tools")
        if isinstance(tools, list) and tools:
            return True
        functions = source.get("functions")
        if isinstance(functions, list) and functions:
            return True
    return False


def extract_tool_names(*sources: dict[str, Any] | None) -> set[str]:
    names: set[str] = set()
    for source in sources:
        if not source:
            continue
        for tool in source.get("tools") or []:
            if isinstance(tool, str) and tool:
                names.add(tool)
                continue
            if not isinstance(tool, dict):
                continue
            if tool.get("name"):
                names.add(str(tool["name"]))
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
        for function in source.get("functions") or []:
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
    return names


def coerce_quality_check(
    check: QualityCheck | Callable[[dict[str, Any]], bool] | None,
) -> QualityCheck | None:
    if check is None:
        return None
    if isinstance(check, _BUILTIN_CHECKS):
        return check
    return CallableCheck(check)


def compose_quality_checks(
    *checks: QualityCheck | Callable[[dict[str, Any]], bool] | None,
) -> QualityCheck | None:
    resolved = [item for item in (coerce_quality_check(check) for check in checks) if item]
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return AllChecks(resolved)
