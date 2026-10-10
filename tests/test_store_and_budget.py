from __future__ import annotations

from pathlib import Path

from freewhirr.budget import BudgetGate
from freewhirr.config import FreewhirrConfig
from freewhirr.models import parse_model
from freewhirr.store import JSONStore, SQLiteStore
from freewhirr.types import Complexity, Outcome


def _record(store, *, outcome: str = Outcome.SUCCESS_FREE.value, cost: float = 0.0) -> None:
    store.record_event(
        request_id="r1",
        winning_model="m",
        models_tried=[{"model": "m", "reason": "success"}],
        outcome=outcome,
        latency_ms=10,
        prompt_tokens=5,
        completion_tokens=5,
        cost_usd=cost,
        baseline_cost_usd=0.01,
        complexity=None,
        policy="paid",
    )


def test_sqlite_and_json_counts(tmp_path: Path) -> None:
    for store in (SQLiteStore(tmp_path / "s.sqlite"), JSONStore(tmp_path / "s.json")):
        _record(store, outcome=Outcome.SUCCESS_FREE.value)
        _record(store, outcome=Outcome.SUCCESS_PAID.value, cost=0.2)
        total, paid = store.success_counts()
        assert total == 2
        assert paid == 1
        assert store.spent_today() == 0.2
        assert store.spent_this_month() == 0.2
        assert len(store.events()) == 2
        store.close()


def test_catalog_first_seen(tmp_path: Path) -> None:
    model = parse_model(
        {"id": "openrouter/horizon-beta", "pricing": {"prompt": "0", "completion": "0"}}
    )
    for store in (SQLiteStore(tmp_path / "c.sqlite"), JSONStore(tmp_path / "c.json")):
        first = store.remember_catalog([model])
        assert first == ["openrouter/horizon-beta"]
        second = store.remember_catalog([model])
        assert second == []
        seen = store.catalog_seen()
        assert seen["openrouter/horizon-beta"]["first_seen"]
        store.close()


def test_share_and_complexity_and_daily_cap(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "b.sqlite")
    config = FreewhirrConfig(
        exhaustion_policy="paid",
        paid_models=["openai/gpt-4o-mini"],
        max_paid_share=0.2,
        paid_only_high_complexity=True,
        max_paid_per_day=1.0,
        max_paid_per_request=0.5,
    )
    gate = BudgetGate(config, store)
    denied = gate.decide(complexity=Complexity.NORMAL, estimated_cost=0.01)
    assert not denied.allowed
    assert "high-complexity" in denied.reason

    allowed = gate.decide(complexity=Complexity.HIGH, estimated_cost=0.01)
    assert allowed.allowed

    _record(store, outcome=Outcome.SUCCESS_PAID.value, cost=0.0)
    share_block = gate.decide(complexity=Complexity.HIGH, estimated_cost=0.01)
    assert not share_block.allowed
    assert "share" in share_block.reason

    store2 = SQLiteStore(tmp_path / "b2.sqlite")
    _record(store2, outcome=Outcome.SUCCESS_PAID.value, cost=0.9)
    gate2 = BudgetGate(
        FreewhirrConfig(
            exhaustion_policy="paid",
            paid_models=["openai/gpt-4o-mini"],
            max_paid_per_day=1.0,
        ),
        store2,
    )
    over = gate2.decide(complexity=None, estimated_cost=0.2)
    assert not over.allowed
    assert "daily" in over.reason
