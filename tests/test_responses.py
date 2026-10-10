from __future__ import annotations

from freewhirr.responses import (
    openai_chunks_to_responses_sse,
    openai_to_responses,
    responses_to_openai,
)


def test_instructions_and_string_input() -> None:
    converted = responses_to_openai(
        {
            "model": "gpt-5",
            "instructions": "Be brief.",
            "input": "hello",
            "max_output_tokens": 32,
        }
    )
    assert converted["messages"][0] == {"role": "system", "content": "Be brief."}
    assert converted["messages"][1] == {"role": "user", "content": "hello"}
    assert converted["max_tokens"] == 32


def test_function_call_roundtrip() -> None:
    converted = responses_to_openai(
        {
            "input": [
                {"role": "user", "content": "lookup x"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "found"},
                {"type": "reasoning", "summary": [{"text": "drop me"}]},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert converted["tool_choice"] == "auto"
    assert converted["tools"][0]["function"]["name"] == "lookup"
    assert converted["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert converted["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "found",
    }
    assert all(item.get("role") != "reasoning" for item in converted["messages"])


def test_openai_to_responses_text_and_usage() -> None:
    raw = {
        "id": "gen-abc",
        "model": "google/gemma-2-9b-it:free",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    payload = openai_to_responses(raw)
    assert payload["object"] == "response"
    assert payload["status"] == "completed"
    assert payload["output"][0]["content"][0]["text"] == "hi"
    assert payload["usage"]["input_tokens"] == 3
    assert payload["usage"]["output_tokens"] == 1


def test_openai_to_responses_function_call() -> None:
    raw = {
        "id": "gen-tool",
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
    }
    payload = openai_to_responses(raw)
    assert payload["output"][0]["type"] == "function_call"
    assert payload["output"][0]["name"] == "lookup"
    assert payload["output"][0]["arguments"] == '{"q":"x"}'


def test_responses_sse_text() -> None:
    chunks = [
        {"id": "s1", "choices": [{"index": 0, "delta": {"content": "Hi"}}]},
        {
            "id": "s1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    ]
    events = "".join(
        openai_chunks_to_responses_sse(iter(chunks), model="freewhirr", response_id="resp_1")
    )
    assert "event: response.created" in events
    assert "event: response.output_text.delta" in events
    assert '"delta":"Hi"' in events
    assert "event: response.completed" in events


def test_responses_sse_function_call() -> None:
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
    events = "".join(openai_chunks_to_responses_sse(iter(chunks), model="freewhirr"))
    assert "function_call" in events
    assert "response.function_call_arguments.delta" in events
    assert "response.completed" in events
