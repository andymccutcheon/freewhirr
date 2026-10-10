"""Thin httpx wrappers for the OpenRouter API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from freewhirr.config import FreewhirrConfig
from freewhirr.errors import FailoverError, OpenRouterError
from freewhirr.retry import classify_failure


def build_headers(config: FreewhirrConfig) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {config.require_api_key()}",
        "Content-Type": "application/json",
        "User-Agent": f"{config.app_title}/0.1.0",
        "HTTP-Referer": config.app_url,
        "X-Title": config.app_title,
        "X-OpenRouter-Title": config.app_title,
    }
    return headers


def provider_preferences(config: FreewhirrConfig) -> dict[str, Any] | None:
    if config.deny_data_collection:
        return {"data_collection": "deny"}
    return None


class OpenRouterTransport:
    def __init__(
        self,
        config: FreewhirrConfig,
        *,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        timeout = httpx.Timeout(config.timeout)
        self._owns_client = client is None
        self._owns_async_client = async_client is None
        self.client = client or httpx.Client(timeout=timeout)
        self.async_client = async_client or httpx.AsyncClient(timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    async def aclose(self) -> None:
        if self._owns_async_client:
            await self.async_client.aclose()
        if self._owns_client:
            self.client.close()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    def list_models(self) -> list[dict[str, Any]]:
        response = self.client.get(self._url("models"), headers=build_headers(self.config))
        return _parse_models_response(response)

    async def alist_models(self) -> list[dict[str, Any]]:
        response = await self.async_client.get(
            self._url("models"), headers=build_headers(self.config)
        )
        return _parse_models_response(response)

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(
            self._url("chat/completions"),
            headers=build_headers(self.config),
            json=payload,
        )
        return _parse_completion_response(response, model=str(payload.get("model") or ""))

    async def acomplete(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.async_client.post(
            self._url("chat/completions"),
            headers=build_headers(self.config),
            json=payload,
        )
        return _parse_completion_response(response, model=str(payload.get("model") or ""))

    def stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        model = str(payload.get("model") or "")
        with self.client.stream(
            "POST",
            self._url("chat/completions"),
            headers=build_headers(self.config),
            json=payload,
        ) as response:
            _raise_for_status(response, model=model)
            yield from iter_sse_payloads(response.iter_lines(), model=model)

    async def astream(self, payload: dict[str, Any]):
        model = str(payload.get("model") or "")
        async with self.async_client.stream(
            "POST",
            self._url("chat/completions"),
            headers=build_headers(self.config),
            json=payload,
        ) as response:
            await _araise_for_status(response, model=model)
            async for chunk in aiter_sse_payloads(response.aiter_lines(), model=model):
                yield chunk

    def generation(self, generation_id: str) -> dict[str, Any]:
        response = self.client.get(
            self._url("generation"),
            headers=build_headers(self.config),
            params={"id": generation_id},
        )
        if response.status_code >= 400:
            return {}
        try:
            data = response.json()
        except json.JSONDecodeError:
            return {}
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            return data["data"]
        return data if isinstance(data, dict) else {}


def _parse_models_response(response: httpx.Response) -> list[dict[str, Any]]:
    if response.status_code >= 400:
        raise OpenRouterError(
            f"Failed to list models ({response.status_code}): {response.text[:300]}",
            status_code=response.status_code,
            body=_safe_json(response),
        )
    data = _safe_json(response)
    if isinstance(data, dict):
        models = data.get("data") or []
        if isinstance(models, list):
            return [item for item in models if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise OpenRouterError("Unexpected models response shape", body=data)


def _parse_completion_response(response: httpx.Response, *, model: str) -> dict[str, Any]:
    _raise_for_status(response, model=model)
    data = _safe_json(response)
    if not isinstance(data, dict):
        raise FailoverError(
            "Upstream returned a non-object completion payload",
            reason="unknown",
            model=model,
            body=data,
        )
    if data.get("error"):
        raise _failover_from_body(model=model, status_code=response.status_code, body=data)
    return data


def _raise_for_status(response: httpx.Response, *, model: str) -> None:
    if response.status_code < 400:
        return
    body = _safe_json(response)
    raise _failover_from_body(model=model, status_code=response.status_code, body=body)


async def _araise_for_status(response: httpx.Response, *, model: str) -> None:
    if response.status_code >= 400:
        # Read the body so classification can see the error payload.
        await response.aread()
        _raise_for_status(response, model=model)


def _failover_from_body(*, model: str, status_code: int | None, body: Any) -> FailoverError:
    message = ""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif error:
            message = str(error)
    if not message:
        message = f"OpenRouter request failed with status {status_code}"
    reason, retryable = classify_failure(status_code=status_code, body=body, message=message)
    return FailoverError(
        message,
        reason=reason.value,
        model=model,
        status_code=status_code,
        body=body,
        retryable=retryable,
    )


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"error": {"message": response.text[:500]}}


def iter_sse_payloads(lines: Iterator[str], *, model: str) -> Iterator[dict[str, Any]]:
    for raw in lines:
        payload = _parse_sse_line(raw)
        if payload is None:
            continue
        if payload is _DONE:
            return
        if payload.get("error"):
            raise _failover_from_body(model=model, status_code=None, body=payload)
        yield payload


async def aiter_sse_payloads(lines, *, model: str):
    async for raw in lines:
        payload = _parse_sse_line(raw)
        if payload is None:
            continue
        if payload is _DONE:
            return
        if payload.get("error"):
            raise _failover_from_body(model=model, status_code=None, body=payload)
        yield payload


_DONE = object()


def _parse_sse_line(raw: str) -> dict[str, Any] | object | None:
    line = raw.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return _DONE
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
