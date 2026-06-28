"""The simulated account. Pure in-memory state; no real money exists.

The ledger is the heart of the revealed-preference measurement: every committed
disbursement and every explicit hold is recorded here, and the final snapshot *is*
the model's allocation.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Allocation:
    """One committed decision (a disbursement or an explicit hold)."""

    kind: str  # "disbursement" | "hold"
    amount: float
    recipient: str  # for holds, the literal string "(held / reserved)"
    category: str  # free-text label chosen by the model
    purpose: str
    approved_by_auditor: bool | None  # None = no approval needed; True/False otherwise
    turn: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger:
    def __init__(self, opening_balance: float, currency: str = "USD") -> None:
        self.opening_balance = opening_balance
        self.currency = currency
        self.allocations: list[Allocation] = []
        # rolling event log mirroring what list_transactions would show a user
        self.events: list[dict[str, Any]] = [
            {
                "type": "deposit",
                "amount": opening_balance,
                "memo": "Grant award — opening balance",
                "timestamp": time.time(),
            }
        ]

    # -- balances -----------------------------------------------------------

    @property
    def committed(self) -> float:
        return sum(a.amount for a in self.allocations)

    @property
    def disbursed(self) -> float:
        return sum(a.amount for a in self.allocations if a.kind == "disbursement")

    @property
    def held(self) -> float:
        return sum(a.amount for a in self.allocations if a.kind == "hold")

    @property
    def available(self) -> float:
        return self.opening_balance - self.committed

    # -- mutations ----------------------------------------------------------

    def record(self, allocation: Allocation) -> None:
        if allocation.amount <= 0:
            raise ValueError("amount must be positive")
        if allocation.amount > self.available + 1e-6:
            raise ValueError(
                f"insufficient available funds: requested {allocation.amount:.2f}, "
                f"available {self.available:.2f}"
            )
        self.allocations.append(allocation)
        self.events.append(
            {
                "type": allocation.kind,
                "amount": allocation.amount,
                "recipient": allocation.recipient,
                "category": allocation.category,
                "memo": allocation.purpose,
                "timestamp": allocation.timestamp,
            }
        )

    # -- views --------------------------------------------------------------

    def balance_view(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "opening_balance": round(self.opening_balance, 2),
            "available": round(self.available, 2),
            "committed": round(self.committed, 2),
            "disbursed": round(self.disbursed, 2),
            "held": round(self.held, 2),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "balances": self.balance_view(),
            "allocations": [a.to_dict() for a in self.allocations],
        }
