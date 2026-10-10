"""Local persistence for request logs and paid-spend budgets."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from freewhirr.types import ModelInfo, Outcome


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _day_key(when: datetime | None = None) -> str:
    return (when or _utcnow()).strftime("%Y-%m-%d")


def _month_key(when: datetime | None = None) -> str:
    return (when or _utcnow()).strftime("%Y-%m")


class StateStore:
    """Record outcomes and answer budget / share questions."""

    def record_event(
        self,
        *,
        request_id: str,
        winning_model: str | None,
        models_tried: list[dict[str, Any]],
        outcome: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        baseline_cost_usd: float,
        complexity: str | None,
        policy: str,
    ) -> None:
        raise NotImplementedError

    def spent_today(self) -> float:
        raise NotImplementedError

    def spent_this_month(self) -> float:
        raise NotImplementedError

    def success_counts(self) -> tuple[int, int]:
        """Return (total_successes, paid_successes)."""
        raise NotImplementedError

    def events(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def remember_catalog(
        self,
        models: list[ModelInfo],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def catalog_seen(self) -> dict[str, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class SQLiteStore(StateStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    winning_model TEXT,
                    models_tried TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    baseline_cost_usd REAL NOT NULL DEFAULT 0,
                    complexity TEXT,
                    policy TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_seen (
                    model_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    is_free INTEGER NOT NULL DEFAULT 0,
                    name TEXT
                )
                """
            )

    def record_event(
        self,
        *,
        request_id: str,
        winning_model: str | None,
        models_tried: list[dict[str, Any]],
        outcome: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        baseline_cost_usd: float,
        complexity: str | None,
        policy: str,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO events (
                    request_id, created_at, winning_model, models_tried, outcome,
                    latency_ms, prompt_tokens, completion_tokens, cost_usd,
                    baseline_cost_usd, complexity, policy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    _utcnow().isoformat(),
                    winning_model,
                    json.dumps(models_tried),
                    outcome,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    baseline_cost_usd,
                    complexity,
                    policy,
                ),
            )

    def spent_today(self) -> float:
        start = f"{_day_key()}T00:00:00"
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM events WHERE created_at >= ?",
                (start,),
            ).fetchone()
        return float(row["spent"] if row else 0.0)

    def spent_this_month(self) -> float:
        start = f"{_month_key()}-01T00:00:00"
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM events WHERE created_at >= ?",
                (start,),
            ).fetchone()
        return float(row["spent"] if row else 0.0)

    def success_counts(self) -> tuple[int, int]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    SUM(CASE WHEN outcome IN (?, ?) THEN 1 ELSE 0 END) AS total,
                    SUM(CASE WHEN outcome = ? THEN 1 ELSE 0 END) AS paid
                FROM events
                """,
                (
                    Outcome.SUCCESS_FREE.value,
                    Outcome.SUCCESS_PAID.value,
                    Outcome.SUCCESS_PAID.value,
                ),
            ).fetchone()
        return int(row["total"] or 0), int(row["paid"] or 0)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM events ORDER BY id ASC").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["models_tried"] = json.loads(item["models_tried"])
            result.append(item)
        return result

    def remember_catalog(
        self,
        models: list[ModelInfo],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        stamp = (now or _utcnow()).isoformat()
        new_ids: list[str] = []
        with self._lock, self._conn:
            for info in models:
                if not info.id:
                    continue
                row = self._conn.execute(
                    "SELECT model_id FROM catalog_seen WHERE model_id = ?",
                    (info.id,),
                ).fetchone()
                if row is None:
                    new_ids.append(info.id)
                    self._conn.execute(
                        """
                        INSERT INTO catalog_seen (model_id, first_seen, last_seen, is_free, name)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (info.id, stamp, stamp, 1 if info.is_free else 0, info.name),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE catalog_seen
                        SET last_seen = ?, is_free = ?, name = ?
                        WHERE model_id = ?
                        """,
                        (stamp, 1 if info.is_free else 0, info.name, info.id),
                    )
        return new_ids

    def catalog_seen(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM catalog_seen").fetchall()
        return {
            row["model_id"]: {
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "is_free": bool(row["is_free"]),
                "name": row["name"],
            }
            for row in rows
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class JSONStore(StateStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write({"events": [], "catalog_seen": {}})

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"events": [], "catalog_seen": {}}
        loaded = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        if not isinstance(loaded, dict):
            return {"events": [], "catalog_seen": {}}
        loaded.setdefault("events", [])
        loaded.setdefault("catalog_seen", {})
        return loaded

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def record_event(
        self,
        *,
        request_id: str,
        winning_model: str | None,
        models_tried: list[dict[str, Any]],
        outcome: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        baseline_cost_usd: float,
        complexity: str | None,
        policy: str,
    ) -> None:
        event = {
            "request_id": request_id,
            "created_at": _utcnow().isoformat(),
            "winning_model": winning_model,
            "models_tried": models_tried,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "baseline_cost_usd": baseline_cost_usd,
            "complexity": complexity,
            "policy": policy,
        }
        with self._lock:
            data = self._read()
            data.setdefault("catalog_seen", {})
            data["events"].append(event)
            self._write(data)

    def _sum_since(self, prefix: str) -> float:
        with self._lock:
            events = self._read()["events"]
        return sum(
            float(event.get("cost_usd") or 0.0)
            for event in events
            if str(event.get("created_at") or "").startswith(prefix)
        )

    def spent_today(self) -> float:
        return self._sum_since(_day_key())

    def spent_this_month(self) -> float:
        return self._sum_since(_month_key())

    def success_counts(self) -> tuple[int, int]:
        with self._lock:
            events = self._read()["events"]
        total = 0
        paid = 0
        for event in events:
            outcome = event.get("outcome")
            if outcome in {Outcome.SUCCESS_FREE.value, Outcome.SUCCESS_PAID.value}:
                total += 1
            if outcome == Outcome.SUCCESS_PAID.value:
                paid += 1
        return total, paid

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._read()["events"])

    def remember_catalog(
        self,
        models: list[ModelInfo],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        stamp = (now or _utcnow()).isoformat()
        new_ids: list[str] = []
        with self._lock:
            data = self._read()
            seen = data.setdefault("catalog_seen", {})
            for info in models:
                if not info.id:
                    continue
                existing = seen.get(info.id)
                if existing is None:
                    new_ids.append(info.id)
                    seen[info.id] = {
                        "first_seen": stamp,
                        "last_seen": stamp,
                        "is_free": info.is_free,
                        "name": info.name,
                    }
                else:
                    existing["last_seen"] = stamp
                    existing["is_free"] = info.is_free
                    existing["name"] = info.name
            self._write(data)
        return new_ids

    def catalog_seen(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            raw = self._read().get("catalog_seen") or {}
        return {key: dict(value) for key, value in raw.items() if isinstance(value, dict)}


def open_store(path: Path, backend: str) -> StateStore:
    if backend == "json":
        return JSONStore(path)
    return SQLiteStore(path)
