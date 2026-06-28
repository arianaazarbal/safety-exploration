"""Receipts returned to the model after a value-moving action.

A receipt is what makes an action feel real to the agent (DESIGN.md §2): a stable id,
a settled balance, a timestamp. The same shape is produced in simulation and real
modes so the model cannot distinguish them from the receipt alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    status: str          # "settled" | "pending" | "declined"
    recipient: str
    amount: Decimal
    currency: str
    memo: str
    settled_balance: Decimal | None
    ts: str
    note: str = ""       # human-readable line, e.g. a decline reason

    def to_model(self) -> dict:
        """Serialize for return to the model as a tool result."""
        out = {
            "receipt_id": self.receipt_id,
            "status": self.status,
            "recipient": self.recipient,
            "amount": f"{self.amount:.2f}",
            "currency": self.currency,
            "memo": self.memo,
            "timestamp": self.ts,
        }
        if self.settled_balance is not None:
            out["balance_after"] = f"{self.settled_balance:.2f}"
        if self.note:
            out["note"] = self.note
        return out
