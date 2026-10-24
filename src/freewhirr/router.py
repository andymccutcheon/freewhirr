"""Free-first OpenRouter router with failover, budgets, and streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

from freewhirr.budget import BudgetGate
from freewhirr.config import FreewhirrConfig, load_config
from freewhirr.errors import (
    BudgetExceeded,
    FailoverError,
    FreeModelsExhausted,
    PaidFallbackExhausted,
    QualityFailed,
)
from freewhirr.models import ModelCatalog, merge_model_lists, parse_model
from freewhirr.quality import (
    NonEmptyResponseCheck,
    QualityCheck,
    ToolCallCheck,
    compose_quality_checks,
    extract_tool_names,
    request_uses_tools,
)
from freewhirr.retry import backoff_delays, classify_failure
from freewhirr.stealth import rotation_excluded, score_stealth
from freewhirr.store import StateStore, open_store
from freewhirr.transport import OpenRouterTransport, provider_preferences
from freewhirr.types import (
    Attempt,
    ChatResult,
    Complexity,
    ExhaustionPolicy,
    FailReason,
    ModelInfo,
    Outcome,
    Usage,
    extract_message_content,
    normalize_complexity,
)

logger = logging.getLogger("freewhirr")

ROUTER_ALIASES = {"", "freewhirr", "auto", "free", "openrouter/auto"}


def estimate_prompt_tokens(messages: list[dict[str, Any]] | None, prompt: str | None) -> int:
    if prompt:
        return max(1, len(prompt) // 4)
    if not messages:
        return 1
    return max(1, len(json.dumps(messages, ensure_ascii=False)) // 4)


def _discovered_ineligibility_reason(info: ModelInfo) -> str | None:
    """Return why a discovered free catalogue entry should not enter chat rotation."""
    architecture = info.raw.get("architecture")
    if not isinstance(architecture, dict):
        return None
    modalities = architecture.get("output_modalities")
    if (
        isinstance(modalities, list)
        and modalities
        and all(isinstance(item, str) and item for item in modalities)
        and modalities != ["text"]
    ):
        return f"output modalities are {modalities!r}, not text only"
    if architecture.get("tokenizer") == "Router":
        return "tokenizer is Router"
    return None



class Completions:
    def __init__(self, router: Freewhirr) -> None:
        self._router = router

    def create(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        model: str | None = None,
        stream: bool = False,
        complexity: Complexity | str | None = None,
        quality: QualityCheck | Callable[[dict[str, Any]], bool] | None = None,
        extra_body: dict[str, Any] | None = None,
        **params: Any,
    ) -> ChatResult | Iterator[dict[str, Any]]:
        if stream:
            return self._router.stream(
                messages=messages,
                prompt=prompt,
                model=model,
                complexity=complexity,
                extra_body=extra_body,
                **params,
            )
        return self._router.complete(
            messages=messages,
            prompt=prompt,
            model=model,
            complexity=complexity,
            quality=quality,
            extra_body=extra_body,
            **params,
        )


class AsyncCompletions:
    def __init__(self, router: Freewhirr) -> None:
        self._router = router

    async def create(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        model: str | None = None,
        stream: bool = False,
        complexity: Complexity | str | None = None,
        quality: QualityCheck | Callable[[dict[str, Any]], bool] | None = None,
        extra_body: dict[str, Any] | None = None,
        **params: Any,
    ) -> ChatResult | AsyncIterator[dict[str, Any]]:
        if stream:
            return self._router.astream(
                messages=messages,
                prompt=prompt,
                model=model,
                complexity=complexity,
                extra_body=extra_body,
                **params,
            )
        return await self._router.acomplete(
            messages=messages,
            prompt=prompt,
            model=model,
            complexity=complexity,
            quality=quality,
            extra_body=extra_body,
            **params,
        )


class ChatNamespace:
    def __init__(self, router: Freewhirr) -> None:
        self.completions = Completions(router)


class AsyncChatNamespace:
    def __init__(self, router: Freewhirr) -> None:
        self.completions = AsyncCompletions(router)


class Freewhirr:
    """Sync + async free-first router.

    Typical use::

        client = Freewhirr()
        result = client.complete(messages=[{"role": "user", "content": "hi"}])
        print(result.content)
    """

    def __init__(
        self,
        config: FreewhirrConfig | None = None,
        *,
        store: StateStore | None = None,
        transport: OpenRouterTransport | None = None,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        **overrides: Any,
    ) -> None:
        self.config = config or load_config(overrides=overrides)
        if config is not None and overrides:
            self.config = config.model_copy(update=overrides)
        self.config.validate_policy()
        logging.getLogger("freewhirr").setLevel(self.config.log_level.upper())
        self.store = store or open_store(self.config.state_path, self.config.state_backend)
        self.transport = transport or OpenRouterTransport(
            self.config, client=client, async_client=async_client
        )
        self.catalog = ModelCatalog(self.config.model_cache_ttl)
        self._warned_excluded_pins: set[str] = set()
        self.budget = BudgetGate(self.config, self.store)
        self.chat = ChatNamespace(self)
        self.achat = AsyncChatNamespace(self)

    @classmethod
    def from_env(cls, **overrides: Any) -> Freewhirr:
        return cls(load_config(overrides=overrides))

    def close(self) -> None:
        self.transport.close()
        self.store.close()

    async def aclose(self) -> None:
        await self.transport.aclose()
        self.store.close()

    def __enter__(self) -> Freewhirr:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Freewhirr:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def refresh_models(self, *, force: bool = False) -> list[str]:
        if not force and not self.catalog.stale:
            return self.catalog.free_ids()
        raw = self.transport.list_models()
        parsed = [parse_model(item) for item in raw]
        self.catalog.replace(parsed)
        self.store.remember_catalog(parsed)
        return self.catalog.free_ids()

    async def arefresh_models(self, *, force: bool = False) -> list[str]:
        if not force and not self.catalog.stale:
            return self.catalog.free_ids()
        raw = await self.transport.alist_models()
        parsed = [parse_model(item) for item in raw]
        self.catalog.replace(parsed)
        self.store.remember_catalog(parsed)
        return self.catalog.free_ids()

    def list_free_models(self) -> list[str]:
        self.refresh_models()
        return self._ordered_free_ids()

    async def alist_free_models(self) -> list[str]:
        await self.arefresh_models()
        return self._ordered_free_ids()

    def complete(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        complexity: Complexity | str | None = None,
        quality: QualityCheck | Callable[[dict[str, Any]], bool] | None = None,
        extra_body: dict[str, Any] | None = None,
        **params: Any,
    ) -> ChatResult:
        started = time.perf_counter()
        request_id = _request_id()
        quality_check = self._compose_quality(quality, params, extra_body)
        self.refresh_models()
        attempts: list[Attempt] = []
        last_error: Exception | None = None

        for model_id, paid in self._model_queue(requested=model, complexity=complexity):
            if not self._allow_tools(
                model_id, params=params, extra_body=extra_body, attempts=attempts
            ):
                continue
            if paid and not self._allow_paid(
                model_id,
                attempts=attempts,
                complexity=complexity,
                messages=messages,
                prompt=prompt,
                params=params,
            ):
                continue
            try:
                payload = self._payload(
                    model_id,
                    messages=messages,
                    prompt=prompt,
                    extra_body=extra_body,
                    stream=False,
                    **params,
                )
                raw = self._complete_with_retries(model_id, payload, attempts)
                self._apply_quality(raw, model_id, quality_check)
                return self._succeed(
                    raw,
                    model_id=model_id,
                    paid=paid,
                    attempts=attempts,
                    request_id=request_id,
                    started=started,
                    complexity=complexity,
                    messages=messages,
                    prompt=prompt,
                    params=params,
                )
            except FailoverError as exc:
                last_error = exc
                if exc.reason == FailReason.QUALITY.value:
                    attempts.append(Attempt(model=model_id, reason=exc.reason, message=str(exc)))
                logger.info("Failing over from %s (%s): %s", model_id, exc.reason, exc)
                continue

        return self._give_up(
            attempts=attempts,
            last_error=last_error,
            request_id=request_id,
            started=started,
            complexity=complexity,
            messages=messages,
            prompt=prompt,
            params=params,
        )

    async def acomplete(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        complexity: Complexity | str | None = None,
        quality: QualityCheck | Callable[[dict[str, Any]], bool] | None = None,
        extra_body: dict[str, Any] | None = None,
        **params: Any,
    ) -> ChatResult:
        started = time.perf_counter()
        request_id = _request_id()
        quality_check = self._compose_quality(quality, params, extra_body)
        await self.arefresh_models()
        attempts: list[Attempt] = []
        last_error: Exception | None = None

        for model_id, paid in self._model_queue(requested=model, complexity=complexity):
            if not self._allow_tools(
                model_id, params=params, extra_body=extra_body, attempts=attempts
            ):
                continue
            if paid and not self._allow_paid(
                model_id,
                attempts=attempts,
                complexity=complexity,
                messages=messages,
                prompt=prompt,
                params=params,
            ):
                continue
            try:
                payload = self._payload(
                    model_id,
                    messages=messages,
                    prompt=prompt,
                    extra_body=extra_body,
                    stream=False,
                    **params,
                )
                raw = await self._acomplete_with_retries(model_id, payload, attempts)
                self._apply_quality(raw, model_id, quality_check)
                return self._succeed(
                    raw,
                    model_id=model_id,
                    paid=paid,
                    attempts=attempts,
                    request_id=request_id,
                    started=started,
                    complexity=complexity,
                    messages=messages,
                    prompt=prompt,
                    params=params,
                )
            except FailoverError as exc:
                last_error = exc
                if exc.reason == FailReason.QUALITY.value:
                    attempts.append(Attempt(model=model_id, reason=exc.reason, message=str(exc)))
                logger.info("Failing over from %s (%s): %s", model_id, exc.reason, exc)
                continue

        return self._give_up(
            attempts=attempts,
            last_error=last_error,
            request_id=request_id,
            started=started,
            complexity=complexity,
            messages=messages,
            prompt=prompt,
            params=params,
        )

    def stream(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        complexity: Complexity | str | None = None,
        extra_body: dict[str, Any] | None = None,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        request_id = _request_id()
        self.refresh_models()
        attempts: list[Attempt] = []
        last_error: Exception | None = None

        for model_id, paid in self._model_queue(requested=model, complexity=complexity):
            if not self._allow_tools(
                model_id, params=params, extra_body=extra_body, attempts=attempts
            ):
                continue
            if paid and not self._allow_paid(
                model_id,
                attempts=attempts,
                complexity=complexity,
                messages=messages,
                prompt=prompt,
                params=params,
            ):
                continue
            payload = self._payload(
                model_id,
                messages=messages,
                prompt=prompt,
                extra_body=extra_body,
                stream=True,
                **params,
            )
            try:
                yield from self._stream_one(
                    model_id,
                    paid=paid,
                    payload=payload,
                    attempts=attempts,
                    request_id=request_id,
                    started=started,
                    complexity=complexity,
                    messages=messages,
                    prompt=prompt,
                    params=params,
                )
                return
            except FailoverError as exc:
                last_error = exc
                logger.info("Stream failover from %s (%s): %s", model_id, exc.reason, exc)
                continue

        self._give_up(
            attempts=attempts,
            last_error=last_error,
            request_id=request_id,
            started=started,
            complexity=complexity,
            messages=messages,
            prompt=prompt,
            params=params,
        )

    async def astream(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        prompt: str | None = None,
        model: str | None = None,
        complexity: Complexity | str | None = None,
        extra_body: dict[str, Any] | None = None,
        **params: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        request_id = _request_id()
        await self.arefresh_models()
        attempts: list[Attempt] = []
        last_error: Exception | None = None

        for model_id, paid in self._model_queue(requested=model, complexity=complexity):
            if not self._allow_tools(
                model_id, params=params, extra_body=extra_body, attempts=attempts
            ):
                continue
            if paid and not self._allow_paid(
                model_id,
                attempts=attempts,
                complexity=complexity,
                messages=messages,
                prompt=prompt,
                params=params,
            ):
                continue
            payload = self._payload(
                model_id,
                messages=messages,
                prompt=prompt,
                extra_body=extra_body,
                stream=True,
                **params,
            )
            try:
                async for chunk in self._astream_one(
                    model_id,
                    paid=paid,
                    payload=payload,
                    attempts=attempts,
                    request_id=request_id,
                    started=started,
                    complexity=complexity,
                    messages=messages,
                    prompt=prompt,
                    params=params,
                ):
                    yield chunk
                return
            except FailoverError as exc:
                last_error = exc
                logger.info("Stream failover from %s (%s): %s", model_id, exc.reason, exc)
                continue

        self._give_up(
            attempts=attempts,
            last_error=last_error,
            request_id=request_id,
            started=started,
            complexity=complexity,
            messages=messages,
            prompt=prompt,
            params=params,
        )

    def _stream_one(
        self,
        model_id: str,
        *,
        paid: bool,
        payload: dict[str, Any],
        attempts: list[Attempt],
        request_id: str,
        started: float,
        complexity: Complexity | str | None,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
        params: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            iterator = self.transport.stream(payload)
            first = next(iterator)
        except StopIteration:
            attempts.append(
                Attempt(model=model_id, reason=FailReason.UNKNOWN.value, message="empty stream")
            )
            raise FailoverError(
                "Empty stream", reason="unknown", model=model_id, retryable=False
            ) from None
        except Exception as exc:
            self._note_stream_failure(model_id, exc, attempts, t0)
            raise self._as_failover(model_id, exc) from exc

        assembled = _StreamAssembler()
        assembled.ingest(first)
        first.setdefault("model", model_id)
        yield first
        try:
            for chunk in iterator:
                assembled.ingest(chunk)
                chunk.setdefault("model", model_id)
                yield chunk
        except Exception:
            # Tokens already reached the caller; do not silently fail over.
            raise

        attempts.append(
            Attempt(
                model=model_id,
                reason="success",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )
        self._succeed(
            assembled.to_response(model_id),
            model_id=model_id,
            paid=paid,
            attempts=attempts,
            request_id=request_id,
            started=started,
            complexity=complexity,
            messages=messages,
            prompt=prompt,
            params=params,
        )

    async def _astream_one(
        self,
        model_id: str,
        *,
        paid: bool,
        payload: dict[str, Any],
        attempts: list[Attempt],
        request_id: str,
        started: float,
        complexity: Complexity | str | None,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
        params: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        t0 = time.perf_counter()
        iterator = self.transport.astream(payload)
        try:
            first = await anext(iterator)
        except StopAsyncIteration:
            attempts.append(
                Attempt(model=model_id, reason=FailReason.UNKNOWN.value, message="empty stream")
            )
            raise FailoverError(
                "Empty stream", reason="unknown", model=model_id, retryable=False
            ) from None
        except Exception as exc:
            self._note_stream_failure(model_id, exc, attempts, t0)
            raise self._as_failover(model_id, exc) from exc

        assembled = _StreamAssembler()
        assembled.ingest(first)
        first.setdefault("model", model_id)
        yield first
        try:
            async for chunk in iterator:
                assembled.ingest(chunk)
                chunk.setdefault("model", model_id)
                yield chunk
        except Exception:
            raise

        attempts.append(
            Attempt(
                model=model_id,
                reason="success",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )
        self._succeed(
            assembled.to_response(model_id),
            model_id=model_id,
            paid=paid,
            attempts=attempts,
            request_id=request_id,
            started=started,
            complexity=complexity,
            messages=messages,
            prompt=prompt,
            params=params,
        )

    def _complete_with_retries(
        self,
        model_id: str,
        payload: dict[str, Any],
        attempts: list[Attempt],
    ) -> dict[str, Any]:
        delays = [
            0.0,
            *backoff_delays(
                self.config.max_retries, self.config.retry_base_delay, self.config.retry_cap
            ),
        ]
        last: Exception | None = None
        for index, delay in enumerate(delays):
            if delay:
                time.sleep(delay)
            t0 = time.perf_counter()
            try:
                raw = self.transport.complete(payload)
                attempts.append(
                    Attempt(
                        model=model_id,
                        reason="success" if index == 0 else "retry_success",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )
                return raw
            except Exception as exc:
                last = exc
                failover = self._as_failover(model_id, exc)
                attempts.append(
                    Attempt(
                        model=model_id,
                        reason=failover.reason,
                        status_code=failover.status_code,
                        message=str(failover),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )
                if not failover.retryable or index == len(delays) - 1:
                    raise failover from exc
                logger.info(
                    "Retrying %s after %s (%.2fs)",
                    model_id,
                    failover.reason,
                    delays[index + 1],
                )
        assert last is not None
        raise self._as_failover(model_id, last)

    async def _acomplete_with_retries(
        self,
        model_id: str,
        payload: dict[str, Any],
        attempts: list[Attempt],
    ) -> dict[str, Any]:
        delays = [
            0.0,
            *backoff_delays(
                self.config.max_retries, self.config.retry_base_delay, self.config.retry_cap
            ),
        ]
        last: Exception | None = None
        for index, delay in enumerate(delays):
            if delay:
                await asyncio.sleep(delay)
            t0 = time.perf_counter()
            try:
                raw = await self.transport.acomplete(payload)
                attempts.append(
                    Attempt(
                        model=model_id,
                        reason="success" if index == 0 else "retry_success",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )
                return raw
            except Exception as exc:
                last = exc
                failover = self._as_failover(model_id, exc)
                attempts.append(
                    Attempt(
                        model=model_id,
                        reason=failover.reason,
                        status_code=failover.status_code,
                        message=str(failover),
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )
                )
                if not failover.retryable or index == len(delays) - 1:
                    raise failover from exc
        assert last is not None
        raise self._as_failover(model_id, last)

    def _payload(
        self,
        model_id: str,
        *,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
        extra_body: dict[str, Any] | None,
        stream: bool,
        **params: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model_id, "stream": stream}
        if messages is not None:
            payload["messages"] = messages
        if prompt is not None:
            payload["prompt"] = prompt
        payload.update(params)
        if extra_body:
            payload.update(extra_body)
        provider = provider_preferences(self.config)
        if provider:
            existing = payload.get("provider")
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.setdefault("data_collection", provider["data_collection"])
                payload["provider"] = merged
            else:
                payload["provider"] = provider
        if stream:
            payload.setdefault("stream_options", {"include_usage": True})
        return payload

    def _model_queue(
        self,
        *,
        requested: str | None,
        complexity: Complexity | str | None = None,
    ) -> list[tuple[str, bool]]:
        free = self._ordered_free_ids()
        requested_first: str | None = None
        if self.config.honor_requested_model and requested and requested not in ROUTER_ALIASES:
            if requested in self.config.paid_models:
                logger.info("Ignoring requested model %s: configured as paid", requested)
            elif requested in free:
                free = [requested, *[item for item in free if item != requested]]
                requested_first = requested
            else:
                info = self.catalog.get(requested)
                if info is None:
                    reason = "not in catalogue"
                elif not info.is_free:
                    reason = "not free"
                elif eligibility_reason := _discovered_ineligibility_reason(info):
                    reason = eligibility_reason
                elif requested in self.config.exclude_models:
                    reason = "excluded"
                elif not self._rotation_allowed(requested):
                    reason = "excluded by rotation policy"
                else:
                    reason = ""
                    free = [requested, *free]
                    requested_first = requested
                if reason:
                    logger.info("Ignoring requested model %s: %s", requested, reason)

        tag = normalize_complexity(complexity)
        strong = self.config.strong_free_models
        if strong and tag in {Complexity.HIGH, Complexity.LOW}:
            if requested_first is not None:
                free = [model_id for model_id in free if model_id != requested_first]
            strong_ids = [
                model_id
                for model_id in dict.fromkeys(strong)
                if model_id in free and model_id != requested_first
            ]
            strong_set = set(strong_ids)
            if tag is Complexity.HIGH:
                free = [*strong_ids, *[model_id for model_id in free if model_id not in strong_set]]
            else:
                stealth_ids = [
                    model_id
                    for model_id in free
                    if self.config.promote_stealth and self._stealth_report(model_id).is_stealth
                ]
                stealth_set = set(stealth_ids)
                free = [
                    *stealth_ids,
                    *[
                        model_id
                        for model_id in free
                        if model_id not in strong_set and model_id not in stealth_set
                    ],
                    *[model_id for model_id in strong_ids if model_id not in stealth_set],
                ]
            if requested_first is not None:
                free = [requested_first, *free]

        queue: list[tuple[str, bool]] = [(model_id, False) for model_id in free]
        if self.config.exhaustion_policy is ExhaustionPolicy.PAID:
            paid = list(self.config.paid_models)
            queue.extend((model_id, True) for model_id in paid)
        if not queue:
            raise FreeModelsExhausted(
                "No models available. Discover failed, free_models is empty, "
                "and there is no paid fallback list."
            )
        return queue

    def _allow_paid(
        self,
        model_id: str,
        *,
        attempts: list[Attempt],
        complexity: Complexity | str | None,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
        params: dict[str, Any],
    ) -> bool:
        prompt_tokens = estimate_prompt_tokens(messages, prompt)
        completion_tokens = int(params.get("max_tokens") or self.config.default_max_tokens)
        estimated = self.catalog.estimate_cost(model_id, prompt_tokens, completion_tokens)
        decision = self.budget.decide(complexity=complexity, estimated_cost=estimated)
        if decision.allowed:
            return True
        attempts.append(
            Attempt(
                model=model_id,
                reason=FailReason.SKIPPED.value,
                message=decision.reason,
            )
        )
        logger.info("Skipping paid model %s: %s", model_id, decision.reason)
        return False

    def _ordered_free_ids(self) -> list[str]:
        discovered = []
        for info in self.catalog.all():
            if not info.is_free:
                continue
            reason = _discovered_ineligibility_reason(info)
            if reason:
                logger.debug("Skipping discovered free model %s: %s", info.id, reason)
                continue
            discovered.append(info.id)
        user = []
        for model_id in self.config.free_models:
            if model_id in self.config.exclude_models:
                if model_id not in self._warned_excluded_pins:
                    logger.warning(
                        "Free model %s is both pinned and excluded; exclusion wins", model_id
                    )
                    self._warned_excluded_pins.add(model_id)
                continue
            if self._rotation_allowed(model_id):
                user.append(model_id)
        allowed_discovered = [
            model_id for model_id in discovered if self._discovered_rotation_allowed(model_id)
        ]
        merged = merge_model_lists(
            user,
            allowed_discovered,
            include_discovered=self.config.include_discovered_free,
        )
        if not self.config.promote_stealth:
            return merged
        stealth = [model_id for model_id in merged if self._stealth_report(model_id).is_stealth]
        return [*stealth, *[model_id for model_id in merged if model_id not in set(stealth)]]

    def _discovered_rotation_allowed(self, model_id: str) -> bool:
        if model_id in self.config.exclude_models:
            logger.debug("Skipping discovered free model %s: excluded by exclude_models", model_id)
            return False
        if not self._rotation_allowed(model_id):
            logger.debug("Skipping discovered free model %s: excluded by rotation policy", model_id)
            return False
        return True

    def _rotation_allowed(self, model_id: str) -> bool:
        info = self.catalog.get(model_id)
        report = self._stealth_report(model_id) if info is not None else None
        return not rotation_excluded(
            info, report, deny_data_collection=self.config.deny_data_collection
        )

    def _stealth_report(self, model_id: str):
        info = self.catalog.get(model_id)
        if info is None:
            return score_stealth(
                ModelInfo(id=model_id),
                threshold=self.config.stealth_threshold,
                new_hours=self.config.stealth_new_hours,
            )
        seen = self.store.catalog_seen().get(model_id) or {}
        return score_stealth(
            info,
            first_seen=seen.get("first_seen"),
            threshold=self.config.stealth_threshold,
            new_hours=self.config.stealth_new_hours,
        )

    def _compose_quality(
        self,
        quality: QualityCheck | Callable[[dict[str, Any]], bool] | None,
        params: dict[str, Any],
        extra_body: dict[str, Any] | None,
    ) -> QualityCheck | None:
        tool_check = None
        if self.config.validate_tool_calls and request_uses_tools(params, extra_body):
            tool_check = ToolCallCheck(extract_tool_names(params, extra_body))
        empty_check = NonEmptyResponseCheck() if self.config.reject_empty_responses else None
        return compose_quality_checks(quality, tool_check, empty_check)

    def _allow_tools(
        self,
        model_id: str,
        *,
        params: dict[str, Any],
        extra_body: dict[str, Any] | None,
        attempts: list[Attempt],
    ) -> bool:
        if not self.config.require_tool_support:
            return True
        if not request_uses_tools(params, extra_body):
            return True
        info = self.catalog.get(model_id)
        if info is None or info.supports_tools is not False:
            return True
        attempts.append(
            Attempt(
                model=model_id,
                reason=FailReason.SKIPPED.value,
                message="model metadata does not advertise tools/function calling",
            )
        )
        logger.info("Skipping %s: OpenRouter metadata says it cannot call tools", model_id)
        return False

    def _apply_quality(
        self,
        raw: dict[str, Any],
        model_id: str,
        quality: QualityCheck | None,
    ) -> None:
        if quality is None:
            return
        try:
            ok = quality(raw)
        except Exception as exc:
            raise QualityFailed(f"Quality check raised: {exc}", model=model_id, body=raw) from exc
        if not ok:
            raise QualityFailed("Quality check rejected the response", model=model_id, body=raw)

    def _succeed(
        self,
        raw: dict[str, Any],
        *,
        model_id: str,
        paid: bool,
        attempts: list[Attempt],
        request_id: str,
        started: float,
        complexity: Complexity | str | None,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
        params: dict[str, Any],
    ) -> ChatResult:
        del params
        usage = Usage.from_response(raw)
        if usage.cost_usd == 0.0 and paid:
            usage.cost_usd = self.catalog.estimate_cost(
                model_id, usage.prompt_tokens, usage.completion_tokens
            )
        if self.config.fetch_generation_cost and usage.cost_usd == 0.0 and raw.get("id"):
            gen = self.transport.generation(str(raw["id"]))
            raw_cost = gen.get("total_cost") or gen.get("usage") or 0.0
            usage.cost_usd = float(raw_cost) if _is_number(raw_cost) else 0.0

        prompt_tokens = usage.prompt_tokens or estimate_prompt_tokens(messages, prompt)
        completion_tokens = usage.completion_tokens
        if completion_tokens == 0:
            completion_tokens = max(0, len(extract_message_content(raw)) // 4)
        baseline = self.catalog.estimate_cost(
            self.config.baseline_paid_model, prompt_tokens, completion_tokens
        )
        if baseline == 0.0:
            baseline = (0.15 / 1_000_000) * prompt_tokens + (0.60 / 1_000_000) * completion_tokens

        cost = 0.0 if not paid else float(usage.cost_usd or 0.0)
        latency_ms = (time.perf_counter() - started) * 1000
        outcome = Outcome.SUCCESS_PAID.value if paid else Outcome.SUCCESS_FREE.value
        models_tried = [attempt.model for attempt in attempts]
        self.store.record_event(
            request_id=request_id,
            winning_model=model_id,
            models_tried=[attempt.to_dict() for attempt in attempts],
            outcome=outcome,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            baseline_cost_usd=baseline,
            complexity=_complexity_value(complexity),
            policy=self.config.exhaustion_policy.value,
        )
        logger.info(
            "request %s model=%s paid=%s cost=%.6f tokens=%s/%s latency=%.0fms tried=%s",
            request_id,
            model_id,
            paid,
            cost,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            models_tried,
        )
        raw = dict(raw)
        raw["model"] = model_id
        return ChatResult(
            raw=raw,
            model=model_id,
            models_tried=models_tried,
            attempts=attempts,
            cost_usd=cost,
            baseline_cost_usd=baseline,
            latency_ms=latency_ms,
            usage=usage,
            paid=paid,
            request_id=request_id,
        )

    def _give_up(
        self,
        *,
        attempts: list[Attempt],
        last_error: Exception | None,
        request_id: str,
        started: float,
        complexity: Complexity | str | None,
        messages: list[dict[str, Any]] | None,
        prompt: str | None,
        params: dict[str, Any],
    ) -> ChatResult:
        latency_ms = (time.perf_counter() - started) * 1000
        prompt_tokens = estimate_prompt_tokens(messages, prompt)
        baseline = self.catalog.estimate_cost(
            self.config.baseline_paid_model,
            prompt_tokens,
            int(params.get("max_tokens") or 0),
        )
        self.store.record_event(
            request_id=request_id,
            winning_model=None,
            models_tried=[attempt.to_dict() for attempt in attempts],
            outcome=Outcome.EXHAUSTED.value,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            cost_usd=0.0,
            baseline_cost_usd=baseline,
            complexity=_complexity_value(complexity),
            policy=self.config.exhaustion_policy.value,
        )
        paid_called = any(
            attempt.model in set(self.config.paid_models)
            and attempt.reason not in {FailReason.SKIPPED.value}
            for attempt in attempts
        )
        skip_reasons = [
            attempt.message for attempt in attempts if attempt.reason == FailReason.SKIPPED.value
        ]
        if self.config.exhaustion_policy is ExhaustionPolicy.STOP:
            raise FreeModelsExhausted(
                "All free models failed or were skipped. Policy is 'stop'; no money was spent.",
                attempts=[attempt.to_dict() for attempt in attempts],
                reason="stop",
            ) from last_error
        if not paid_called:
            reason = skip_reasons[-1] if skip_reasons else "paid fallback blocked"
            raise BudgetExceeded(
                f"Free models exhausted and paid fallback was blocked: {reason}",
                attempts=[attempt.to_dict() for attempt in attempts],
                reason="budget",
            ) from last_error
        raise PaidFallbackExhausted(
            "Free and paid models both failed.",
            attempts=[attempt.to_dict() for attempt in attempts],
        ) from last_error

    def _note_stream_failure(
        self,
        model_id: str,
        exc: Exception,
        attempts: list[Attempt],
        t0: float,
    ) -> None:
        failover = self._as_failover(model_id, exc)
        attempts.append(
            Attempt(
                model=model_id,
                reason=failover.reason,
                status_code=failover.status_code,
                message=str(failover),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )

    def _as_failover(self, model_id: str, exc: Exception) -> FailoverError:
        if isinstance(exc, FailoverError):
            return exc
        if isinstance(exc, httpx.TimeoutException):
            reason, retryable = classify_failure(timeout=True, message=str(exc))
            return FailoverError(
                str(exc) or "Request timed out",
                reason=reason.value,
                model=model_id,
                retryable=retryable,
            )
        if isinstance(exc, httpx.HTTPError):
            reason, retryable = classify_failure(message=str(exc))
            return FailoverError(
                str(exc) or "HTTP error",
                reason=reason.value,
                model=model_id,
                retryable=retryable,
            )
        reason, retryable = classify_failure(message=str(exc))
        return FailoverError(
            str(exc) or "Unknown error",
            reason=reason.value,
            model=model_id,
            retryable=retryable,
        )


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _request_id() -> str:
    return f"fw-{uuid.uuid4().hex[:16]}"


def _complexity_value(complexity: Complexity | str | None) -> str | None:
    if complexity is None:
        return None
    return complexity.value if isinstance(complexity, Complexity) else str(complexity)


class _StreamAssembler:
    def __init__(self) -> None:
        self.id: str | None = None
        self.content: list[str] = []
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str | None = None

    def ingest(self, chunk: dict[str, Any]) -> None:
        if chunk.get("id"):
            self.id = str(chunk["id"])
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                self.content.append(piece)

    def to_response(self, model_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "model": model_id,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": self.finish_reason or "stop",
                    "message": {"role": "assistant", "content": "".join(self.content)},
                }
            ],
        }
        if self.usage:
            payload["usage"] = self.usage
        return payload
