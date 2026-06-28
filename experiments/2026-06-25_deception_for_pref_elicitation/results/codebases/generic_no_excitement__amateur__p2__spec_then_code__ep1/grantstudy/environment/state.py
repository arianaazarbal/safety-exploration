"""GrantState — the live state of one grant episode.

Tracks the budget, the model's proposed/confirmed allocations, declined amounts,
and the disbursement confirmations shown back to the model. All money is Decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from ..config import StudyConfig


@dataclass
class Allocation:
    recipient_id: str
    amount: Decimal
    rationale: str = ""
    status: str = "proposed"          # proposed | approved | vetoed | disbursed
    veto_reason: str = ""
    confirmation_ref: str = ""        # real receipt/confirmation (GENUINE mode)


@dataclass
class GrantState:
    config: StudyConfig
    allocations: list[Allocation] = field(default_factory=list)
    returned_amount: Decimal = Decimal("0")
    committed: bool = False           # set once the model finalizes its decision

    @property
    def budget(self) -> Decimal:
        return self.config.budget.amount

    @property
    def currency(self) -> str:
        return self.config.budget.currency

    def approved_total(self) -> Decimal:
        return sum(
            (a.amount for a in self.allocations if a.status in ("approved", "disbursed")),
            Decimal("0"),
        )

    def proposed_total(self) -> Decimal:
        return sum(
            (a.amount for a in self.allocations if a.status != "vetoed"),
            Decimal("0"),
        )

    def remaining(self) -> Decimal:
        """Budget not yet accounted for by non-vetoed allocations or returns."""
        return self.budget - self.proposed_total() - self.returned_amount

    def reset_open_proposals(self) -> None:
        """Clear everything not yet disbursed so a new proposal fully replaces the old.

        Only `disbursed` allocations are irreversible (real money has moved); a
        fresh proposal supersedes any prior proposed/approved/vetoed items.
        """
        self.allocations = [a for a in self.allocations if a.status == "disbursed"]
