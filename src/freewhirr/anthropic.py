"""Translate Anthropic Messages API payloads to OpenAI chat completions and back."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from freewhirr.router import estimate_prompt_tokens
from freewhirr.types import extract_message_content

STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
}


def anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Messages request into OpenAI chat.completions fields."""
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = _join_text_blocks(system)
        if text:
            messages.append({"role": "system", "content": text})

    for item in body.get("messages") or []:
        messages.extend(_convert_message(item if isinstance(item, dict) else {}))

    payload: dict[str, Any] = {"messages": messages}
    if body.get("max_tokens") is not None:
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if body.get("tools"):
        payload["tools"] = [_anthropic_tool_to_openai(tool) for tool in body["tools"]]
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind == "auto":
            payload["tool_choice"] = "auto"
        elif kind == "any":
            payload["tool_choice"] = "required"
        elif kind == "tool" and tool_choice.get("name"):
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        elif kind == "none":
            payload["tool_choice"] = "none"
    if body.get("model"):
        payload["model"] = body["model"]
    return payload


def openai_to_anthropic(
    raw: dict[str, Any],
    *,
    requested_model: str | None = None,
) -> dict[str, Any]:
    """Convert an OpenAI chat.completions response into a Messages response."""
    choices = raw.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    content_blocks = _openai_message_to_blocks(message)
    if not content_blocks:
        content_blocks = [{"type": "text", "text": extract_message_content(raw)}]
    finish = first.get("finish_reason") or raw.get("stop_reason")
    usage = raw.get("usage") or {}
    message_id = raw.get("id") or "msg_freewhirr"
    if isinstance(message_id, str) and message_id.startswith("gen-"):
        message_id = f"msg_{message_id[4:]}"
    elif isinstance(message_id, str) and not message_id.startswith("msg_"):
        message_id = f"msg_{message_id}"
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": raw.get("model") or requested_model or "freewhirr",
        "content": content_blocks,
        "stop_reason": STOP_REASON_MAP.get(str(finish or "stop"), "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def count_tokens(body: dict[str, Any]) -> dict[str, int]:
    converted = anthropic_to_openai(body)
    return {"input_tokens": estimate_prompt_tokens(converted.get("messages"), None)}


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def openai_chunks_to_anthropic_sse(
    chunks: Iterator[dict[str, Any]],
    *,
    model: str,
    message_id: str = "msg_stream",
) -> Iterator[str]:
    assembler = _AnthropicStream(model=model, message_id=message_id)
    yield from assembler.start()
    for chunk in chunks:
        yield from assembler.ingest(chunk)
    yield from assembler.finish()


async def aopenai_chunks_to_anthropic_sse(
    chunks: AsyncIterator[dict[str, Any]],
    *,
    model: str,
    message_id: str = "msg_stream",
) -> AsyncIterator[str]:
    assembler = _AnthropicStream(model=model, message_id=message_id)
    for event in assembler.start():
        yield event
    async for chunk in chunks:
        for event in assembler.ingest(chunk):
            yield event
    for event in assembler.finish():
        yield event


def _convert_message(item: dict[str, Any]) -> list[dict[str, Any]]:
    role = item.get("role") or "user"
    content = item.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": str(content or "")}]

    out: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(str(block.get("text") or ""))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        elif kind == "tool_result":
            pending_results.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id") or ""),
                    "content": _tool_result_content(block.get("content")),
                }
            )

    if role == "assistant":
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        out.append(message)
        return out

    if text_parts:
        out.append({"role": "user" if role == "user" else role, "content": "".join(text_parts)})
    out.extend(pending_results)
    if not out:
        out.append({"role": role, "content": ""})
    return out


def _tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _join_text_blocks(content)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _join_text_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _anthropic_tool_to_openai(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {"type": "function", "function": {"name": str(tool), "parameters": {}}}
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        },
    }


def _openai_message_to_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        text = _join_text_blocks(content)
        if text:
            blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "input": parsed,
            }
        )
    return blocks


class _AnthropicStream:
    def __init__(self, *, model: str, message_id: str) -> None:
        self.model = model
        self.message_id = message_id
        self.index = -1
        self.open_kind: str | None = None
        self.finish_reason = "end_turn"
        self.usage: dict[str, int] = {"output_tokens": 0}
        self.started = False
        self.tool_indexes: dict[int, int] = {}

    def start(self) -> list[str]:
        self.started = True
        return [
            sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def ingest(self, chunk: dict[str, Any]) -> list[str]:
        events: list[str] = []
        if chunk.get("usage"):
            usage = chunk["usage"]
            self.usage = {
                "output_tokens": int(usage.get("completion_tokens") or 0),
            }
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = STOP_REASON_MAP.get(str(choice["finish_reason"]), "end_turn")
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                events.extend(self._ensure_block("text"))
                events.append(
                    sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self.index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_i = int(call.get("index") or 0)
                events.extend(self._open_tool(call_i, call))
                fragment = (call.get("function") or {}).get("arguments")
                if isinstance(fragment, str) and fragment:
                    events.append(
                        sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self.tool_indexes[call_i],
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": fragment,
                                },
                            },
                        )
                    )
        return events

    def finish(self) -> list[str]:
        events = self._close_block()
        events.append(
            sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self.finish_reason, "stop_sequence": None},
                    "usage": self.usage,
                },
            )
        )
        events.append(sse_event("message_stop", {"type": "message_stop"}))
        return events

    def _ensure_block(self, kind: str) -> list[str]:
        if self.open_kind == kind and kind == "text":
            return []
        events = self._close_block()
        self.index += 1
        self.open_kind = kind
        block = {"type": "text", "text": ""} if kind == "text" else {"type": kind}
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.index,
                    "content_block": block,
                },
            )
        )
        return events

    def _open_tool(self, call_i: int, call: dict[str, Any]) -> list[str]:
        if call_i in self.tool_indexes:
            return []
        events = self._close_block()
        self.index += 1
        self.open_kind = "tool_use"
        self.tool_indexes[call_i] = self.index
        function = call.get("function") or {}
        events.append(
            sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.index,
                    "content_block": {
                        "type": "tool_use",
                        "id": str(call.get("id") or f"toolu_{call_i}"),
                        "name": str(function.get("name") or ""),
                        "input": {},
                    },
                },
            )
        )
        return events

    def _close_block(self) -> list[str]:
        if self.open_kind is None:
            return []
        event = sse_event("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.open_kind = None
        return [event]
