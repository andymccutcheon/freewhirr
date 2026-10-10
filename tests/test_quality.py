from __future__ import annotations

from freewhirr.quality import (
    CallableCheck,
    JsonSchemaCheck,
    MinLengthCheck,
    ToolCallCheck,
    compose_quality_checks,
    extract_tool_names,
    request_uses_tools,
)


def _payload(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_min_length() -> None:
    check = MinLengthCheck(5)
    assert check(_payload("hello world"))
    assert not check(_payload("hi"))


def test_json_schema_accepts_valid() -> None:
    check = JsonSchemaCheck(
        {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    )
    assert check(_payload('{"ok": true}'))
    assert check(_payload('```json\n{"ok": false}\n```'))
    assert not check(_payload('{"ok": "nope"}'))
    assert not check(_payload("not json"))


def test_callable_check() -> None:
    check = CallableCheck(lambda payload: "yes" in str(payload))
    assert check(_payload("yes"))
    assert not check(_payload("nope"))


def _tool_payload(name: str = "lookup", arguments: str = '{"q":"x"}') -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    }


def test_tool_call_check() -> None:
    check = ToolCallCheck({"lookup"})
    assert check(_tool_payload())
    assert not check(_tool_payload(arguments="{"))
    assert not check(_tool_payload(name="other"))
    assert check(_payload("plain text is fine"))
    assert not check(
        {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": []}}]}
    )


def test_request_uses_tools_and_names() -> None:
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    assert request_uses_tools({"tools": tools})
    assert not request_uses_tools({"tools": []}, None)
    assert extract_tool_names({"tools": tools}) == {"lookup"}


def test_compose_quality_checks() -> None:
    combined = compose_quality_checks(MinLengthCheck(2), ToolCallCheck({"lookup"}))
    assert combined is not None
    assert combined(_payload("hello"))
    assert not combined(_payload("x"))
