"""Guardrails that decide whether a paid escalation is allowed."""

from __future__ import annotations

from dataclasses import dataclass

from freewhirr.config import FreewhirrConfig
from freewhirr.store import StateStore
from freewhirr.types import Complexity, normalize_complexity


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str = ""


class BudgetGate:
    def __init__(self, config: FreewhirrConfig, store: StateStore) -> None:
        self.config = config
        self.store = store

    def decide(
        self,
        *,
        complexity: Complexity | str | None,
        estimated_cost: float,
    ) -> BudgetDecision:
        if self.config.paid_only_high_complexity and not _is_high(complexity):
            return BudgetDecision(
                False,
                "paid escalation is limited to high-complexity requests",
            )
        if not self._share_allows():
            return BudgetDecision(False, "max paid share would be exceeded")
        if (
            self.config.max_paid_per_request is not None
            and estimated_cost > self.config.max_paid_per_request
        ):
            return BudgetDecision(
                False,
                f"estimated ${estimated_cost:.6f} exceeds max_paid_per_request "
                f"${self.config.max_paid_per_request:.6f}",
            )
        if self.config.max_paid_per_day is not None:
            spent = self.store.spent_today()
            if spent + estimated_cost > self.config.max_paid_per_day:
                return BudgetDecision(
                    False,
                    f"daily spend ${spent:.6f} + estimate ${estimated_cost:.6f} "
                    f"exceeds max_paid_per_day ${self.config.max_paid_per_day:.6f}",
                )
        if self.config.max_paid_per_month is not None:
            spent = self.store.spent_this_month()
            if spent + estimated_cost > self.config.max_paid_per_month:
                return BudgetDecision(
                    False,
                    f"monthly spend ${spent:.6f} + estimate ${estimated_cost:.6f} "
                    f"exceeds max_paid_per_month ${self.config.max_paid_per_month:.6f}",
                )
        return BudgetDecision(True)

    def _share_allows(self) -> bool:
        cap = self.config.max_paid_share
        if cap is None:
            return True
        total, paid = self.store.success_counts()
        if total == 0:
            return True
        return (paid + 1) / (total + 1) <= cap + 1e-12


def _is_high(complexity: Complexity | str | None) -> bool:
    return normalize_complexity(complexity) is Complexity.HIGH
