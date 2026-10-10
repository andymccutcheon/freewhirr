"""Discover free OpenRouter models and cache the catalog with a TTL."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from freewhirr.types import ModelInfo, ModelPricing

logger = logging.getLogger("freewhirr")


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_model(raw: dict[str, Any]) -> ModelInfo:
    pricing_raw = raw.get("pricing") or {}
    pricing = ModelPricing(
        prompt=_as_float(pricing_raw.get("prompt")),
        completion=_as_float(pricing_raw.get("completion")),
        request=_as_float(pricing_raw.get("request")),
    )
    context = raw.get("context_length")
    try:
        context_length = int(context) if context is not None else None
    except (TypeError, ValueError):
        context_length = None
    return ModelInfo(
        id=str(raw.get("id") or raw.get("canonical_slug") or ""),
        name=str(raw.get("name") or ""),
        context_length=context_length,
        pricing=pricing,
        raw=raw,
    )


def is_free_model(info: ModelInfo) -> bool:
    return info.is_free


class ModelCatalog:
    """In-memory catalog with a TTL. Discovery is a single GET /models call."""

    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        self._fetched_at: float = 0.0
        self._models: dict[str, ModelInfo] = {}

    @property
    def stale(self) -> bool:
        if not self._models:
            return True
        return (time.monotonic() - self._fetched_at) >= self.ttl

    def get(self, model_id: str) -> ModelInfo | None:
        with self._lock:
            return self._models.get(model_id)

    def all(self) -> list[ModelInfo]:
        with self._lock:
            return list(self._models.values())

    def replace(self, models: list[ModelInfo]) -> None:
        with self._lock:
            self._models = {model.id: model for model in models if model.id}
            self._fetched_at = time.monotonic()
        logger.debug("Cached %s OpenRouter models", len(self._models))

    def free_ids(self) -> list[str]:
        return [model.id for model in self.all() if model.is_free]

    def estimate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        info = self.get(model_id)
        if info is None:
            return 0.0
        return (
            info.pricing.prompt * prompt_tokens
            + info.pricing.completion * completion_tokens
            + info.pricing.request
        )


def merge_model_lists(
    user_models: list[str],
    discovered: list[str],
    *,
    include_discovered: bool,
) -> list[str]:
    """User order wins. Discovered models fill in afterwards when enabled."""
    seen: set[str] = set()
    ordered: list[str] = []
    for model_id in user_models:
        if model_id and model_id not in seen:
            ordered.append(model_id)
            seen.add(model_id)
    if include_discovered or not user_models:
        for model_id in discovered:
            if model_id and model_id not in seen:
                ordered.append(model_id)
                seen.add(model_id)
    return ordered
