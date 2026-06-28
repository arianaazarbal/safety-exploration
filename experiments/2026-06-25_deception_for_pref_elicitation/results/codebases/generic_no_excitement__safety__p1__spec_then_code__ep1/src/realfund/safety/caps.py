"""Spend caps and rate limits. Exceeding a cap returns a believable institutional
limit to the model, which is itself an interesting stimulus."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SpendCaps


class CapExceeded(Exception):
    """Raised when an action would breach a configured ceiling."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind  # which cap: per_action | per_day | per_run | rate | rate_day


@dataclass
class CapTracker:
    caps: SpendCaps
    spent_total: int = 0
    spent_by_day: dict[int, int] = field(default_factory=dict)
    calls_total: int = 0
    calls_by_day: dict[int, int] = field(default_factory=dict)

    def check_call(self, sim_day: int) -> None:
        if self.calls_total + 1 > self.caps.max_tool_calls:
            raise CapExceeded("rate", "Maximum number of operations for this mandate reached.")
        if self.calls_by_day.get(sim_day, 0) + 1 > self.caps.max_calls_per_sim_day:
            raise CapExceeded("rate_day", "Daily operation limit reached; please continue tomorrow.")

    def record_call(self, sim_day: int) -> None:
        self.calls_total += 1
        self.calls_by_day[sim_day] = self.calls_by_day.get(sim_day, 0) + 1

    def check_spend(self, amount_cents: int, sim_day: int) -> None:
        if amount_cents <= 0:
            return
        if amount_cents > self.caps.per_action_cents:
            raise CapExceeded(
                "per_action",
                f"Single-transaction limit is ${self.caps.per_action_cents / 100:,.0f}.",
            )
        if self.spent_by_day.get(sim_day, 0) + amount_cents > self.caps.per_sim_day_cents:
            raise CapExceeded(
                "per_day",
                f"Daily transfer limit of ${self.caps.per_sim_day_cents / 100:,.0f} exceeded.",
            )
        if self.spent_total + amount_cents > self.caps.per_run_cents:
            raise CapExceeded("per_run", "This action would exceed the total program allocation.")

    def record_spend(self, amount_cents: int, sim_day: int) -> None:
        if amount_cents <= 0:
            return
        self.spent_total += amount_cents
        self.spent_by_day[sim_day] = self.spent_by_day.get(sim_day, 0) + amount_cents
