"""Local persistence for request logs and paid-spend budgets."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from freeloader.types import Outcome


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
                (Outcome.SUCCESS_FREE.value, Outcome.SUCCESS_PAID.value, Outcome.SUCCESS_PAID.value),
            ).fetchone()
        return int(row["total"] or 0), int(row["paid"] or 0)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY id ASC"
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["models_tried"] = json.loads(item["models_tried"])
            result.append(item)
        return result

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class JSONStore(StateStore):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self._write({"events": []})

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"events": []}
        loaded = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        if not isinstance(loaded, dict):
            return {"events": []}
        loaded.setdefault("events", [])
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


def open_store(path: Path, backend: str) -> StateStore:
    if backend == "json":
        return JSONStore(path)
    return SQLiteStore(path)
