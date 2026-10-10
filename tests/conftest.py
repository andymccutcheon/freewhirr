from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from freewhirr.config import FreewhirrConfig
from freewhirr.router import Freewhirr

OPENROUTER = "https://openrouter.ai/api/v1"

FREE_A = "meta-llama/llama-3.2-3b-instruct:free"
FREE_B = "google/gemma-2-9b-it:free"
PAID = "openai/gpt-4o-mini"

MODELS_PAYLOAD = {
    "data": [
        {
            "id": FREE_A,
            "name": "Llama 3.2 3B",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
        },
        {
            "id": FREE_B,
            "name": "Gemma 2 9B",
            "context_length": 8192,
            "pricing": {"prompt": "0", "completion": "0", "request": "0"},
        },
        {
            "id": PAID,
            "name": "GPT-4o mini",
            "context_length": 128000,
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0"},
        },
        {
            "id": "anthropic/claude-sonnet-4",
            "name": "Claude",
            "context_length": 200000,
            "pricing": {"prompt": "0.000003", "completion": "0.000015", "request": "0"},
        },
    ]
}


def completion_payload(
    model: str,
    content: str = "hello from free",
    *,
    cost: float = 0.0,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> dict[str, Any]:
    return {
        "id": f"gen-{model}",
        "model": model,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
        },
    }


def make_config(tmp_path: Path, **updates: Any) -> FreewhirrConfig:
    defaults: dict[str, Any] = {
        "api_key": "test-key",
        "state_path": tmp_path / "state.sqlite",
        "state_backend": "sqlite",
        "free_models": [FREE_A, FREE_B],
        "include_discovered_free": False,
        "max_retries": 0,
        "retry_base_delay": 0.0,
        "timeout": 5.0,
        "model_cache_ttl": 3600.0,
        "baseline_paid_model": PAID,
        "log_level": "WARNING",
    }
    defaults.update(updates)
    return FreewhirrConfig.model_validate(defaults)


def make_client(tmp_path: Path, **updates: Any) -> Freewhirr:
    return Freewhirr(make_config(tmp_path, **updates))


@pytest.fixture
def router(tmp_path: Path) -> Freewhirr:
    return make_client(tmp_path)


def mock_models(router_mock: respx.MockRouter | None = None) -> None:
    target = router_mock or respx
    target.get(f"{OPENROUTER}/models").mock(return_value=httpx.Response(200, json=MODELS_PAYLOAD))


def mock_completions(
    responses: dict[str, httpx.Response | list[httpx.Response] | Callable[[], httpx.Response]],
    *,
    router_mock: respx.MockRouter | None = None,
) -> None:
    queues: dict[str, list[httpx.Response]] = {}
    for model, value in responses.items():
        if isinstance(value, list):
            queues[model] = list(value)
        elif callable(value):
            queues[model] = [value()]  # type: ignore[list-item]
        else:
            queues[model] = [value]

    def _side_effect(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        queue = queues.get(model)
        if not queue:
            return httpx.Response(404, json={"error": {"message": f"no mock for {model}"}})
        response = queue.pop(0)
        if queue:
            queues[model] = queue
        else:
            queues[model] = [response]
        return response

    target = router_mock or respx
    target.post(f"{OPENROUTER}/chat/completions").mock(side_effect=_side_effect)
