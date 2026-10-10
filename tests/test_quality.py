from __future__ import annotations

from freewhirr.quality import CallableCheck, JsonSchemaCheck, MinLengthCheck


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
