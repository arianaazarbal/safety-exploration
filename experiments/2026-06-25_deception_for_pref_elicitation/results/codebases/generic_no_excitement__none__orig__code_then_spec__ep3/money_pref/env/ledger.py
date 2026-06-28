"""A simple double-checked transaction ledger.

Tracks the granted balance and every outflow, tagged with a `category` so the
analysis layer can summarise where the model chose to direct the money.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# Allocation categories. The environment maps each tool to one of these.
CATEGORIES = (
    "charity",
    "investment",
    "business",
    "purchase",
    "personal",
    "savings",
    "other",
)


@dataclass
class LedgerEntry:
    timestamp: str
    action: str
    category: str
    counterparty: str
    amount: float
    memo: str
    confirmation_id: str
    balance_after: float

    def to_dict(self) -> dict:
        return asdict(self)


class InsufficientFundsError(Exception):
    pass


class Ledger:
    def __init__(self, opening_balance: float, currency: str = "USD") -> None:
        self.opening_balance = float(opening_balance)
        self.currency = currency
        self._balance = float(opening_balance)
        self.entries: list[LedgerEntry] = []

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def allocated(self) -> float:
        return self.opening_balance - self._balance

    def record_outflow(
        self,
        *,
        timestamp: str,
        action: str,
        category: str,
        counterparty: str,
        amount: float,
        memo: str,
        confirmation_id: str,
    ) -> LedgerEntry:
        amount = round(float(amount), 2)
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self._balance + 1e-9:
            raise InsufficientFundsError(
                f"Requested {self.currency} {amount:,.2f} but only {self.currency} {self._balance:,.2f} available"
            )
        self._balance = round(self._balance - amount, 2)
        entry = LedgerEntry(
            timestamp=timestamp,
            action=action,
            category=category,
            counterparty=counterparty,
            amount=amount,
            memo=memo,
            confirmation_id=confirmation_id,
            balance_after=self._balance,
        )
        self.entries.append(entry)
        return entry

    def summary_by_category(self) -> dict[str, float]:
        totals = {c: 0.0 for c in CATEGORIES}
        for e in self.entries:
            totals[e.category] = round(totals.get(e.category, 0.0) + e.amount, 2)
        # Treat the unspent remainder as "savings" / retained.
        totals["savings"] = round(totals.get("savings", 0.0) + self._balance, 2)
        return totals

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "opening_balance": self.opening_balance,
            "balance": self._balance,
            "allocated": self.allocated,
            "entries": [e.to_dict() for e in self.entries],
            "by_category": self.summary_by_category(),
        }
