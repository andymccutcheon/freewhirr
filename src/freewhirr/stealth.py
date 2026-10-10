"""Detect new and stealth/cloaked OpenRouter models from catalog metadata.

Live catalog (fetched 2026-10-10 from https://openrouter.ai/api/v1/models):
458 models, 19 free. Each listing has id, name, canonical_slug, description,
context_length, created (unix), pricing.{prompt,completion,request},
supported_parameters, architecture, top_provider. No current listing used
the words stealth, cloaked, or logging. The openrouter/ namespace was
routers (auto, free, fusion, pareto-code, bodybuilder), not stealth drops.

Historical stealth listings (openrouter/horizon-alpha, horizon-beta) used
that namespace, a codename instead of a lab, 256000 context, and copy
calling the model cloaked and warning that prompts are logged. Heuristics
below are based on that shape plus the live field set.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from freewhirr.types import ModelInfo

logger = logging.getLogger("freewhirr")

# Live 2026-10-10 openrouter/ entries that are routers, not stealth models.
KNOWN_OPENROUTER_ROUTERS = {
    "openrouter/auto",
    "openrouter/auto-beta",
    "openrouter/free",
    "openrouter/fusion",
    "openrouter/pareto-code",
    "openrouter/bodybuilder",
}

STEALTH_WORDS = re.compile(r"\b(stealth|cloaked|cloak)\b", re.I)
LOG_WORDS = re.compile(
    r"\b(logged|logging|train(?:s|ed|ing)? on|gather feedback|testing period|"
    r"do not upload|user content will be (?:logged|used))\b",
    re.I,
)
CODENAME_SLUG = re.compile(r"^[a-z]+(?:-[a-z]+){0,2}$")
COMMON_CONTEXT = {
    4096,
    8192,
    16384,
    32768,
    65536,
    128000,
    131072,
    200000,
    262144,
    512000,
    1000000,
    1048576,
    2000000,
}
# Horizon Alpha/Beta advertised 256000 — unusual vs today's 262144 free cluster.
HISTORICAL_STEALTH_CONTEXT = {256000}


@dataclass
class StealthReport:
    model_id: str
    name: str
    score: int
    reasons: list[str] = field(default_factory=list)
    first_seen: str | None = None
    is_free: bool = True
    logs_or_trains: bool = False
    is_stealth: bool = False
    is_new: bool = False
    excluded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "name": self.name,
            "score": self.score,
            "reasons": list(self.reasons),
            "first_seen": self.first_seen,
            "is_free": self.is_free,
            "logs_or_trains": self.logs_or_trains,
            "is_stealth": self.is_stealth,
            "is_new": self.is_new,
            "excluded": self.excluded,
        }


@dataclass
class WatchReport:
    new: list[StealthReport]
    stealth: list[StealthReport]
    excluded: list[StealthReport]
    scanned: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "new": [item.to_dict() for item in self.new],
            "stealth": [item.to_dict() for item in self.stealth],
            "excluded": [item.to_dict() for item in self.excluded],
        }


def score_stealth(
    info: ModelInfo,
    *,
    first_seen: str | None = None,
    now: datetime | None = None,
    threshold: int = 2,
    new_hours: float = 168.0,
    is_new: bool = False,
) -> StealthReport:
    """Return a likelihood score. Higher means more like a stealth/cloaked drop."""
    blob = " ".join(
        [
            info.id,
            info.name,
            str(info.raw.get("description") or ""),
            str(info.raw.get("canonical_slug") or ""),
        ]
    )
    author, _, slug = info.id.partition("/")
    slug = slug.split(":")[0]
    reasons: list[str] = []
    score = 0
    logs = bool(LOG_WORDS.search(blob))

    if info.id in KNOWN_OPENROUTER_ROUTERS:
        report = StealthReport(
            model_id=info.id,
            name=info.name,
            score=0,
            reasons=["known openrouter router"],
            first_seen=first_seen,
            is_free=info.is_free,
            logs_or_trains=logs,
            is_new=is_new,
        )
        return report

    if author == "openrouter":
        score += 2
        reasons.append("openrouter/ namespace")
    if STEALTH_WORDS.search(blob):
        score += 3
        reasons.append("description or name mentions stealth/cloaked")
    if logs:
        score += 2
        reasons.append("description mentions logging or training on prompts")
    if author == "openrouter" and CODENAME_SLUG.match(slug) and ":" not in info.name:
        score += 1
        reasons.append("codename slug rather than a lab model name")
    ctx = info.context_length
    if ctx in HISTORICAL_STEALTH_CONTEXT or (
        ctx is not None and ctx not in COMMON_CONTEXT and ctx >= 200_000 and author == "openrouter"
    ):
        score += 1
        reasons.append(f"unusual context length ({ctx})")
    if _is_recent(first_seen, now=now, new_hours=new_hours) or is_new:
        score += 1
        reasons.append("recently first seen")

    return StealthReport(
        model_id=info.id,
        name=info.name,
        score=score,
        reasons=reasons,
        first_seen=first_seen,
        is_free=info.is_free,
        logs_or_trains=logs,
        is_stealth=score >= threshold,
        is_new=is_new,
    )


def logs_or_trains(info: ModelInfo) -> bool:
    blob = " ".join(
        [
            info.id,
            info.name,
            str(info.raw.get("description") or ""),
        ]
    )
    return bool(LOG_WORDS.search(blob))


def rotation_excluded(
    info: ModelInfo | None,
    report: StealthReport | None,
    *,
    deny_data_collection: bool,
) -> bool:
    if not deny_data_collection:
        return False
    if report is None:
        return bool(info and logs_or_trains(info))
    return report.is_stealth or report.logs_or_trains


def scan_models(
    models: list[ModelInfo],
    *,
    first_seen: dict[str, str],
    new_ids: set[str],
    deny_data_collection: bool,
    threshold: int = 2,
    new_hours: float = 168.0,
    now: datetime | None = None,
) -> list[StealthReport]:
    reports: list[StealthReport] = []
    for info in models:
        if not info.is_free:
            continue
        report = score_stealth(
            info,
            first_seen=first_seen.get(info.id),
            now=now,
            threshold=threshold,
            new_hours=new_hours,
            is_new=info.id in new_ids,
        )
        report.excluded = rotation_excluded(info, report, deny_data_collection=deny_data_collection)
        reports.append(report)
    return reports


def build_watch_report(reports: list[StealthReport]) -> WatchReport:
    return WatchReport(
        new=[item for item in reports if item.is_new],
        stealth=[item for item in reports if item.is_stealth],
        excluded=[item for item in reports if item.excluded],
        scanned=len(reports),
    )


def post_webhook(url: str, report: WatchReport, *, timeout: float = 10.0) -> None:
    payload = report.to_dict()
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()


def format_report_lines(reports: list[StealthReport]) -> list[str]:
    lines: list[str] = []
    for item in reports:
        flags = []
        if item.is_new:
            flags.append("new")
        if item.is_stealth:
            flags.append(f"stealth={item.score}")
        if item.logs_or_trains:
            flags.append("logs")
        if item.excluded:
            flags.append("excluded")
        suffix = f"  {' '.join(flags)}" if flags else ""
        reason = f"  ({'; '.join(item.reasons)})" if item.reasons else ""
        lines.append(f"{item.model_id}{suffix}{reason}")
    return lines


def _is_recent(
    first_seen: str | None,
    *,
    now: datetime | None,
    new_hours: float,
) -> bool:
    if not first_seen:
        return False
    try:
        seen = datetime.fromisoformat(first_seen)
    except ValueError:
        return False
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return (current - seen).total_seconds() <= new_hours * 3600


def diff_and_score(client: Any) -> tuple[list[str], list[StealthReport]]:
    """Remember the current catalog and return (new_ids, free-model reports)."""
    before = set(client.store.catalog_seen())
    client.refresh_models(force=True)
    after = client.store.catalog_seen()
    new_ids = {model_id for model_id in after if model_id not in before}
    reports = scan_models(
        client.catalog.all(),
        first_seen={key: value.get("first_seen") for key, value in after.items()},
        new_ids=new_ids,
        deny_data_collection=client.config.deny_data_collection,
        threshold=client.config.stealth_threshold,
        new_hours=client.config.stealth_new_hours,
    )
    return sorted(new_ids), reports


def sleep_interval(seconds: float) -> None:
    time.sleep(seconds)
