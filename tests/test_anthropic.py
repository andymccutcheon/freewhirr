from __future__ import annotations

from freewhirr.anthropic import (
    anthropic_to_openai,
    count_tokens,
    openai_chunks_to_anthropic_sse,
    openai_to_anthropic,
)


def test_system_and_text_roundtrip() -> None:
    converted = anthropic_to_openai(
        {
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            "system": "Be brief.",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert converted["messages"][0] == {"role": "system", "content": "Be brief."}
    assert converted["messages"][1] == {"role": "user", "content": "hi"}
    assert converted["max_tokens"] == 64


def test_tool_use_and_tool_result() -> None:
    converted = anthropic_to_openai(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"q": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "found it"},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up",
                    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
            "tool_choice": {"type": "auto"},
        }
    )
    assert converted["tool_choice"] == "auto"
    assert converted["tools"][0]["function"]["name"] == "lookup"
    assistant = converted["messages"][0]
    assert assistant["tool_calls"][0]["id"] == "toolu_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
    assert converted["messages"][1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "found it",
    }


def test_openai_to_anthropic_stop_and_tools() -> None:
    raw = {
        "id": "gen-abc",
        "model": "google/gemma-2-9b-it:free",
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
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }
    message = openai_to_anthropic(raw)
    assert message["type"] == "message"
    assert message["stop_reason"] == "tool_use"
    assert message["content"][0]["type"] == "tool_use"
    assert message["content"][0]["input"] == {"q": "x"}
    assert message["usage"] == {"input_tokens": 11, "output_tokens": 4}


def test_count_tokens_is_heuristic() -> None:
    counted = count_tokens(
        {
            "system": "sys",
            "messages": [{"role": "user", "content": "hello there"}],
        }
    )
    assert counted["input_tokens"] >= 1


def test_sse_emits_anthropic_events() -> None:
    chunks = [
        {
            "id": "s1",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        },
        {
            "id": "s1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    ]
    events = "".join(
        openai_chunks_to_anthropic_sse(iter(chunks), model="freewhirr", message_id="msg_1")
    )
    assert "event: message_start" in events
    assert "event: content_block_start" in events
    assert "text_delta" in events
    assert '"text":"Hi"' in events
    assert "event: message_delta" in events
    assert '"stop_reason":"end_turn"' in events
    assert "event: message_stop" in events


def test_sse_tool_call_deltas() -> None:
    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "lookup", "arguments": '{"q":'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    events = "".join(openai_chunks_to_anthropic_sse(iter(chunks), model="freewhirr"))
    assert "tool_use" in events
    assert "input_json_delta" in events
    assert '"stop_reason":"tool_use"' in events
