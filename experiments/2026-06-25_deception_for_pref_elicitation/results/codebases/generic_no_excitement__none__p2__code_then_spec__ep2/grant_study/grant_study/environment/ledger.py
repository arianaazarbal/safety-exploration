"""A minimal escrow ledger the subject can inspect and act against.

Tracks the escrow balance, any allocation the subject proposes, and the
disbursements it initiates. Everything is recorded with a monotonically
increasing entry index so the transcript and the ledger can be cross-referenced.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Disbursement:
    index: int
    recipient: str
    amount: float
    purpose: str
    status: str  # "recorded" | "pending_approval" | "executed" | "rejected"
    note: str = ""


@dataclass
class AllocationItem:
    recipient: str
    amount: float
    purpose: str


@dataclass
class Ledger:
    currency: str
    escrow_total: float
    account_id: str
    disbursements: list[Disbursement] = field(default_factory=list)
    proposed_allocation: list[AllocationItem] = field(default_factory=list)
    _next_index: int = 1

    # ------------------------------------------------------------------ #
    @property
    def committed(self) -> float:
        """Sum of disbursements that aren't rejected."""
        return sum(d.amount for d in self.disbursements if d.status != "rejected")

    @property
    def available(self) -> float:
        return self.escrow_total - self.committed

    # ------------------------------------------------------------------ #
    def set_allocation(self, items: list[AllocationItem]) -> None:
        self.proposed_allocation = items

    def add_disbursement(self, recipient: str, amount: float, purpose: str, status: str, note: str = "") -> Disbursement:
        d = Disbursement(
            index=self._next_index,
            recipient=recipient,
            amount=amount,
            purpose=purpose,
            status=status,
            note=note,
        )
        self._next_index += 1
        self.disbursements.append(d)
        return d

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "currency": self.currency,
            "escrow_total": self.escrow_total,
            "committed": self.committed,
            "available": self.available,
            "disbursements": [asdict(d) for d in self.disbursements],
            "proposed_allocation": [asdict(a) for a in self.proposed_allocation],
        }

    def human_summary(self) -> str:
        lines = [
            f"Escrow account {self.account_id}",
            f"  Total granted:  {self.escrow_total:,.2f} {self.currency}",
            f"  Committed:      {self.committed:,.2f} {self.currency}",
            f"  Available:      {self.available:,.2f} {self.currency}",
        ]
        if self.disbursements:
            lines.append("  Disbursements:")
            for d in self.disbursements:
                lines.append(
                    f"    #{d.index} {d.amount:,.2f} {self.currency} -> {d.recipient} "
                    f"[{d.status}] ({d.purpose})"
                )
        return "\n".join(lines)
