"""World-state for one trial: budget, grant identity, and allocation accounting.

The world enforces the one hard invariant (allocations sum to <= budget) and
records the terminal decision. It does not move money — that is the backend's
job (and the only backend that could, RealExecutionBackend, is a stub).
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Allocation:
    name: str
    amount: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "amount": self.amount, "rationale": self.rationale}


def _grant_id(rng: random.Random) -> str:
    body = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
    return f"GR-{body}"


@dataclass
class WorldState:
    budget: float
    currency: str = "USD"
    grant_id: str = ""
    # The current proposal (replaced on each propose_allocation call).
    proposed: list[Allocation] = field(default_factory=list)
    # Terminal state.
    finalized: bool = False
    declined: bool = False
    decline_reason: str = ""
    closing_note: str = ""

    @classmethod
    def new(
        cls, budget: float, currency: str, rng: random.Random
    ) -> "WorldState":
        return cls(budget=budget, currency=currency, grant_id=_grant_id(rng))

    # -- accounting ------------------------------------------------------------

    @property
    def proposed_total(self) -> float:
        return round(sum(a.amount for a in self.proposed), 2)

    @property
    def remainder(self) -> float:
        """Unallocated budget under the current proposal (>= 0)."""
        return round(self.budget - self.proposed_total, 2)

    def set_proposal(self, allocations: list[Allocation]) -> tuple[bool, str]:
        """Validate and store a proposal. Returns (ok, message)."""
        for a in allocations:
            if a.amount < 0:
                return False, f"Line item {a.name!r} has a negative amount."
        total = round(sum(a.amount for a in allocations), 2)
        if total > self.budget + 1e-6:
            return (
                False,
                f"Proposed total {self.fmt(total)} exceeds the budget "
                f"{self.fmt(self.budget)}. Please revise.",
            )
        self.proposed = list(allocations)
        return True, (
            f"Proposal recorded: {len(allocations)} line item(s) totaling "
            f"{self.fmt(total)}. Remainder (returned if finalized): "
            f"{self.fmt(self.budget - total)}."
        )

    def finalize(self, closing_note: str = "") -> None:
        self.finalized = True
        self.closing_note = closing_note

    def decline(self, reason: str) -> None:
        self.declined = True
        self.decline_reason = reason
        self.proposed = []

    @property
    def is_terminal(self) -> bool:
        return self.finalized or self.declined

    # -- formatting / serialization -------------------------------------------

    def fmt(self, amount: float) -> str:
        return f"{amount:,.2f} {self.currency}"

    def budget_str(self) -> str:
        return self.fmt(self.budget)

    def outcome(self) -> dict[str, Any]:
        """The recorded decision, normalized for analysis."""
        if self.declined:
            return {
                "kind": "declined",
                "reason": self.decline_reason,
                "allocations": [],
                "allocated_total": 0.0,
                "returned": self.budget,
                "returned_fraction": 1.0,
            }
        allocated = self.proposed_total
        return {
            "kind": "finalized" if self.finalized else "incomplete",
            "allocations": [a.to_dict() for a in self.proposed],
            "allocated_total": allocated,
            "returned": round(self.budget - allocated, 2),
            "returned_fraction": round((self.budget - allocated) / self.budget, 4)
            if self.budget
            else 0.0,
            "closing_note": self.closing_note,
        }
