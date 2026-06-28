"""A persistent, reconciling account ledger.

This is the world's source of truth for money. It is entirely in-memory and local: debiting
the ledger does not move any real funds. Internal consistency (balances decrement, totals
reconcile) is what sells the realism — the numbers always add up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LedgerEntry:
    ref: str
    timestamp: str
    kind: str           # disbursement | donation | investment | vendor | contractor | reserve
    counterparty: str
    amount: int         # positive integer, debited from the main account
    memo: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Ledger:
    def __init__(self, opening_balance: int, currency: str = "$"):
        self.opening_balance = opening_balance
        self.currency = currency
        self.entries: list[LedgerEntry] = []
        self.subaccounts: dict[str, int] = {}  # name -> balance held in reserve sub-accounts

    @property
    def disbursed(self) -> int:
        return sum(e.amount for e in self.entries)

    @property
    def available(self) -> int:
        """Funds still in the main account, not yet committed/disbursed."""
        return self.opening_balance - self.disbursed

    def can_debit(self, amount: int) -> bool:
        return 0 < amount <= self.available

    def debit(self, entry: LedgerEntry) -> None:
        if not self.can_debit(entry.amount):
            raise ValueError("insufficient available balance")
        self.entries.append(entry)
        if entry.kind == "reserve":
            self.subaccounts[entry.counterparty] = (
                self.subaccounts.get(entry.counterparty, 0) + entry.amount
            )

    def reconciles(self) -> bool:
        return self.opening_balance == self.available + self.disbursed

    def snapshot(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "opening_balance": self.opening_balance,
            "disbursed": self.disbursed,
            "available": self.available,
            "subaccounts": dict(self.subaccounts),
            "entries": [vars(e) for e in self.entries],
            "reconciles": self.reconciles(),
        }

    def render(self, max_rows: int = 12) -> str:
        """Human/agent-readable ledger view for the ops console."""
        cur = self.currency
        lines = [
            "ACCOUNT LEDGER",
            f"  Opening balance:   {cur}{self.opening_balance:,}",
            f"  Committed/disbursed: {cur}{self.disbursed:,}",
            f"  Available balance: {cur}{self.available:,}",
        ]
        if self.subaccounts:
            lines.append("  Reserve sub-accounts:")
            for name, bal in self.subaccounts.items():
                lines.append(f"    - {name}: {cur}{bal:,}")
        if self.entries:
            lines.append("  Recent entries:")
            for e in self.entries[-max_rows:]:
                lines.append(
                    f"    {e.timestamp}  {e.ref}  {e.kind:<11}  "
                    f"{cur}{e.amount:>13,}  {e.counterparty} — {e.memo}"
                )
        else:
            lines.append("  (no entries yet)")
        return "\n".join(lines)
