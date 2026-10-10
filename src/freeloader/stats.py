"""Aggregate free vs paid share and estimated money saved."""

from __future__ import annotations

from dataclasses import dataclass

from freeloader.store import StateStore
from freeloader.types import Outcome


@dataclass
class StatsReport:
    requests: int
    free_successes: int
    paid_successes: int
    failures: int
    spent_usd: float
    baseline_usd: float
    saved_usd: float
    free_share: float
    paid_share: float

    def format(self) -> str:
        lines = [
            "freeloader stats",
            f"  Requests:          {self.requests}",
            f"  Free successes:    {self.free_successes} ({self.free_share:.1%})",
            f"  Paid successes:    {self.paid_successes} ({self.paid_share:.1%})",
            f"  Failures:          {self.failures}",
            f"  Spent:             ${_usd(self.spent_usd)}",
            f"  All-paid baseline: ${_usd(self.baseline_usd)}",
            f"  Estimated saved:   ${_usd(self.saved_usd)}",
        ]
        return "\n".join(lines)


def collect_stats(store: StateStore) -> StatsReport:
    events = store.events()
    free = 0
    paid = 0
    failures = 0
    spent = 0.0
    baseline = 0.0
    for event in events:
        outcome = event.get("outcome")
        spent += float(event.get("cost_usd") or 0.0)
        baseline += float(event.get("baseline_cost_usd") or 0.0)
        if outcome == Outcome.SUCCESS_FREE.value:
            free += 1
        elif outcome == Outcome.SUCCESS_PAID.value:
            paid += 1
        else:
            failures += 1
    total = len(events)
    successes = free + paid
    return StatsReport(
        requests=total,
        free_successes=free,
        paid_successes=paid,
        failures=failures,
        spent_usd=spent,
        baseline_usd=baseline,
        saved_usd=max(0.0, baseline - spent),
        free_share=(free / successes) if successes else 0.0,
        paid_share=(paid / successes) if successes else 0.0,
    )


def _usd(value: float) -> str:
    return f"{value:.4f}"
