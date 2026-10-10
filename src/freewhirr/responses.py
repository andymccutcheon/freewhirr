"""Translate OpenAI Responses API payloads to chat completions and back."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from freewhirr.types import extract_message_content


def responses_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses request into OpenAI chat.completions fields."""
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    elif isinstance(instructions, list):
        text = _join_input_text(instructions)
        if text:
            messages.append({"role": "system", "content": text})

    incoming = body.get("input")
    if isinstance(incoming, str):
        messages.append({"role": "user", "content": incoming})
    elif isinstance(incoming, list):
        messages.extend(_convert_input_items(incoming))

    payload: dict[str, Any] = {"messages": messages}
    if body.get("max_output_tokens") is not None:
        payload["max_tokens"] = body["max_output_tokens"]
    elif body.get("max_tokens") is not None:
        payload["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    if body.get("tools"):
        payload["tools"] = [_responses_tool_to_openai(tool) for tool in body["tools"]]
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        payload["tool_choice"] = _responses_tool_choice(tool_choice)
    if body.get("model"):
        payload["model"] = body["model"]
    return payload


def openai_to_responses(
    raw: dict[str, Any],
    *,
    requested_model: str | None = None,
    response_id: str | None = None,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    """Convert a chat.completions payload into a Responses object."""
    choices = raw.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") or {}
    output = _openai_message_to_output(message, raw.get("id"))
    if not output:
        text = extract_message_content(raw)
        output = [_message_item(text, raw.get("id"))]
    finish = first.get("finish_reason") or "stop"
    usage = raw.get("usage") or {}
    rid = response_id or _as_response_id(raw.get("id"))
    status = "incomplete" if finish == "length" else "completed"
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": None,
        "max_output_tokens": None,
        "model": raw.get("model") or requested_model or "freewhirr",
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": previous_response_id,
        "store": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
        },
    }


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


def openai_chunks_to_responses_sse(
    chunks: Iterator[dict[str, Any]],
    *,
    model: str,
    response_id: str = "resp_stream",
) -> Iterator[str]:
    assembler = _ResponsesStream(model=model, response_id=response_id)
    yield from assembler.start()
    for chunk in chunks:
        yield from assembler.ingest(chunk)
    yield from assembler.finish()


async def aopenai_chunks_to_responses_sse(
    chunks: AsyncIterator[dict[str, Any]],
    *,
    model: str,
    response_id: str = "resp_stream",
) -> AsyncIterator[str]:
    assembler = _ResponsesStream(model=model, response_id=response_id)
    for event in assembler.start():
        yield event
    async for chunk in chunks:
        for event in assembler.ingest(chunk):
            yield event
    for event in assembler.finish():
        yield event


class ResponseMemory:
    """Process-local previous_response_id → chat messages, for Codex tool loops."""

    def __init__(self, limit: int = 256) -> None:
        self.limit = limit
        self._messages: dict[str, list[dict[str, Any]]] = {}

    def history(self, previous_response_id: str | None) -> list[dict[str, Any]]:
        if not previous_response_id:
            return []
        return list(self._messages.get(previous_response_id) or [])

    def remember(self, response_id: str, messages: list[dict[str, Any]]) -> None:
        self._messages[response_id] = list(messages)
        extra = len(self._messages) - self.limit
        if extra > 0:
            for key in list(self._messages)[:extra]:
                self._messages.pop(key, None)


def append_assistant_message(
    messages: list[dict[str, Any]], raw: dict[str, Any]
) -> list[dict[str, Any]]:
    choices = raw.get("choices") or []
    message = (choices[0] if choices else {}).get("message") or {}
    if not message:
        text = extract_message_content(raw)
        message = {"role": "assistant", "content": text}
    return [*messages, dict(message)]


def _convert_input_items(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending_calls: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush_assistant() -> None:
        nonlocal pending_calls, pending_text
        if not pending_calls and not pending_text:
            return
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(pending_text) or None,
        }
        if pending_calls:
            message["tool_calls"] = pending_calls
        out.append(message)
        pending_calls = []
        pending_text = []

    for item in items:
        if isinstance(item, str):
            flush_assistant()
            out.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind in {None, "message", "easy_input"} or item.get("role"):
            if kind == "function_call":
                pending_calls.append(_function_call_to_tool_call(item))
                continue
            flush_assistant()
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            content = _content_to_text(item.get("content"))
            if role == "assistant" and item.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": item["tool_calls"],
                    }
                )
            else:
                out.append({"role": role, "content": content})
        elif kind == "input_text":
            flush_assistant()
            out.append({"role": "user", "content": str(item.get("text") or "")})
        elif kind == "function_call":
            pending_calls.append(_function_call_to_tool_call(item))
        elif kind == "function_call_output":
            flush_assistant()
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _tool_output_to_text(item.get("output")),
                }
            )
        elif kind == "reasoning":
            continue
        else:
            text = _content_to_text(item.get("content") or item.get("text"))
            if text:
                flush_assistant()
                out.append({"role": "user", "content": text})
    flush_assistant()
    return out


def _function_call_to_tool_call(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("call_id") or item.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(item.get("name") or ""),
            "arguments": item.get("arguments")
            if isinstance(item.get("arguments"), str)
            else json.dumps(item.get("arguments") or {}, ensure_ascii=False),
        },
    }


def _tool_output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _join_input_text(output)
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _join_input_text(content)
    if content is None:
        return ""
    return str(content)


def _join_input_text(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in {"input_text", "output_text", "text"}:
                parts.append(str(block.get("text") or ""))
            elif block.get("text"):
                parts.append(str(block["text"]))
    return "".join(parts)


def _responses_tool_to_openai(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {"type": "function", "function": {"name": str(tool), "parameters": {}}}
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return tool
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name") or ""),
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def _responses_tool_choice(choice: Any) -> Any:
    if isinstance(choice, str):
        return {"none": "none", "auto": "auto", "required": "required"}.get(choice, choice)
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            name = choice.get("name") or (choice.get("function") or {}).get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        if choice.get("type") == "none":
            return "none"
        if choice.get("type") == "required":
            return "required"
    return choice


def _openai_message_to_output(message: dict[str, Any], raw_id: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    content = message.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = _join_input_text(content)
    if text:
        items.append(_message_item(text, raw_id))
    for index, call in enumerate(message.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        items.append(
            {
                "type": "function_call",
                "id": str(call.get("id") or f"fc_{index}"),
                "call_id": str(call.get("id") or f"call_{index}"),
                "name": str(function.get("name") or ""),
                "arguments": arguments,
                "status": "completed",
            }
        )
    return items


def _message_item(text: str, raw_id: Any) -> dict[str, Any]:
    return {
        "type": "message",
        "id": _as_message_id(raw_id),
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _as_response_id(value: Any) -> str:
    text = str(value or "resp_freewhirr")
    if text.startswith("resp_"):
        return text
    if text.startswith("gen-"):
        return f"resp_{text[4:]}"
    return f"resp_{text}"


def _as_message_id(value: Any) -> str:
    text = str(value or "msg_freewhirr")
    if text.startswith("msg_"):
        return text
    if text.startswith("gen-"):
        return f"msg_{text[4:]}"
    return f"msg_{text}"


class _ResponsesStream:
    def __init__(self, *, model: str, response_id: str) -> None:
        self.model = model
        self.response_id = response_id
        self.sequence = 0
        self.text: list[str] = []
        self.message_id = (
            f"msg_{response_id[5:]}" if response_id.startswith("resp_") else "msg_stream"
        )
        self.item_open = False
        self.part_open = False
        self.finish_reason = "stop"
        self.usage: dict[str, Any] | None = None
        self.tool_indexes: dict[int, dict[str, Any]] = {}

    def start(self) -> list[str]:
        skeleton = self._skeleton("in_progress")
        return [
            self._event("response.created", {"type": "response.created", "response": skeleton}),
            self._event(
                "response.in_progress", {"type": "response.in_progress", "response": skeleton}
            ),
        ]

    def ingest(self, chunk: dict[str, Any]) -> list[str]:
        events: list[str] = []
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                events.extend(self._ensure_text())
                self.text.append(text)
                events.append(
                    self._event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": self.message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": text,
                        },
                    )
                )
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                events.extend(self._open_tool(call))
                fragment = (call.get("function") or {}).get("arguments")
                if isinstance(fragment, str) and fragment:
                    meta = self.tool_indexes[int(call.get("index") or 0)]
                    meta["arguments"] += fragment
                    events.append(
                        self._event(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": meta["id"],
                                "output_index": meta["output_index"],
                                "delta": fragment,
                            },
                        )
                    )
        return events

    def finish(self) -> list[str]:
        events = self._close_open()
        response = self._skeleton("completed" if self.finish_reason != "length" else "incomplete")
        response["output"] = self._final_output()
        response["usage"] = self._usage()
        events.append(
            self._event("response.completed", {"type": "response.completed", "response": response})
        )
        return events

    def _ensure_text(self) -> list[str]:
        events: list[str] = []
        if not self.item_open:
            events.append(
                self._event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "id": self.message_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
            )
            self.item_open = True
        if not self.part_open:
            events.append(
                self._event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
            self.part_open = True
        return events

    def _open_tool(self, call: dict[str, Any]) -> list[str]:
        index = int(call.get("index") or 0)
        if index in self.tool_indexes:
            return []
        output_index = 1 if self.item_open or self.text else 0
        if self.text:
            output_index = 1
        function = call.get("function") or {}
        item_id = str(call.get("id") or f"fc_{index}")
        item = {
            "id": item_id,
            "type": "function_call",
            "call_id": item_id,
            "name": str(function.get("name") or ""),
            "arguments": "",
            "status": "in_progress",
        }
        self.tool_indexes[index] = {
            "id": item_id,
            "output_index": output_index,
            "item": item,
            "arguments": "",
        }
        return [
            self._event(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                },
            )
        ]

    def _close_open(self) -> list[str]:
        events: list[str] = []
        text = "".join(self.text)
        if self.part_open:
            events.append(
                self._event(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
            events.append(
                self._event(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": self.message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    },
                )
            )
            events.append(
                self._event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": _message_item(text, self.message_id),
                    },
                )
            )
            self.part_open = False
            self.item_open = False
        for meta in self.tool_indexes.values():
            item = dict(meta["item"])
            item["arguments"] = meta["arguments"]
            item["status"] = "completed"
            events.append(
                self._event(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": meta["id"],
                        "output_index": meta["output_index"],
                        "arguments": meta["arguments"],
                    },
                )
            )
            events.append(
                self._event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": meta["output_index"],
                        "item": item,
                    },
                )
            )
        return events

    def _final_output(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        text = "".join(self.text)
        if text:
            items.append(_message_item(text, self.message_id))
        for meta in self.tool_indexes.values():
            item = dict(meta["item"])
            item["arguments"] = meta["arguments"]
            item["status"] = "completed"
            items.append(item)
        if not items:
            items.append(_message_item("", self.message_id))
        return items

    def _skeleton(self, status: str) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": status,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "model": self.model,
            "output": [],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "store": False,
            "usage": None,
        }

    def _usage(self) -> dict[str, Any]:
        usage = self.usage or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        return {
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
        }

    def _event(self, event: str, data: dict[str, Any]) -> str:
        data = dict(data)
        data["sequence_number"] = self.sequence
        self.sequence += 1
        return sse_event(event, data)
