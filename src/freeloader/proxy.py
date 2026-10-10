"""OpenAI-compatible HTTP proxy so any language can point ``base_url`` here."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from freeloader.errors import (
    BudgetExceeded,
    ConfigError,
    FreeModelsExhausted,
    FreeloaderError,
    PaidFallbackExhausted,
)
from freeloader.router import Freeloader
from freeloader.stats import collect_stats
from freeloader.types import Complexity


def create_app(client: Freeloader | None = None) -> FastAPI:
    app = FastAPI(
        title="freeloader",
        description="Free-first OpenRouter proxy with an OpenAI-compatible API.",
        version="0.1.0",
    )
    app.state.client = client

    @app.middleware("http")
    async def bind_client(request: Request, call_next):  # type: ignore[no-untyped-def]
        if app.state.client is None:
            try:
                app.state.client = Freeloader.from_env()
            except ConfigError as exc:
                return JSONResponse({"error": {"message": str(exc), "type": "config_error"}}, 500)
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        router: Freeloader = app.state.client
        await router.arefresh_models()
        free = await router.alist_free_models()
        models = [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "openrouter",
                "created": 0,
            }
            for model_id in free
        ]
        if router.config.exhaustion_policy.value == "paid":
            models.extend(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "openrouter",
                    "created": 0,
                }
                for model_id in router.config.paid_models
            )
        models.insert(
            0,
            {"id": "freeloader", "object": "model", "owned_by": "freeloader", "created": 0},
        )
        return {"object": "list", "data": models}

    @app.get("/v1/stats")
    async def stats() -> dict[str, Any]:
        router: Freeloader = app.state.client
        report = collect_stats(router.store)
        return report.__dict__

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_freeloader_complexity: str | None = Header(default=None),
    ) -> Any:
        router: Freeloader = app.state.client
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        stream = bool(body.pop("stream", False))
        complexity = _complexity_from(request, x_freeloader_complexity, body)
        model = body.pop("model", None)
        messages = body.pop("messages", None)
        prompt = body.pop("prompt", None)
        extra_body = {key: value for key, value in body.items() if value is not None}

        if stream:
            return StreamingResponse(
                _stream_response(
                    router,
                    messages=messages,
                    prompt=prompt,
                    model=model,
                    complexity=complexity,
                    extra_body=extra_body,
                ),
                media_type="text/event-stream",
            )

        try:
            result = await router.acomplete(
                messages=messages,
                prompt=prompt,
                model=model,
                complexity=complexity,
                extra_body=extra_body,
            )
        except FreeloaderError as exc:
            raise _http_error(exc) from exc
        payload = result.to_openai()
        payload.setdefault("object", "chat.completion")
        return JSONResponse(payload)

    return app


async def _stream_response(
    router: Freeloader,
    *,
    messages: Any,
    prompt: Any,
    model: Any,
    complexity: Complexity | str | None,
    extra_body: dict[str, Any],
) -> AsyncIterator[str]:
    try:
        async for chunk in router.astream(
            messages=messages,
            prompt=prompt,
            model=model,
            complexity=complexity,
            extra_body=extra_body,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except FreeloaderError as exc:
        error = {
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
            }
        }
        yield f"data: {json.dumps(error)}\n\n"
        yield "data: [DONE]\n\n"


def _complexity_from(
    request: Request,
    header_value: str | None,
    body: dict[str, Any],
) -> str | None:
    if header_value:
        return header_value
    query = request.query_params.get("complexity")
    if query:
        return query
    extra = body.pop("freeloader", None)
    if isinstance(extra, dict) and extra.get("complexity"):
        return str(extra["complexity"])
    return None


def _http_error(exc: FreeloaderError) -> HTTPException:
    status = 402 if isinstance(exc, (FreeModelsExhausted, BudgetExceeded, PaidFallbackExhausted)) else 502
    return HTTPException(
        status_code=status,
        detail={
            "message": str(exc),
            "type": type(exc).__name__,
            "attempts": getattr(exc, "attempts", []),
        },
    )


def run(host: str = "127.0.0.1", port: int = 8787, client: Freeloader | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(client), host=host, port=port)
