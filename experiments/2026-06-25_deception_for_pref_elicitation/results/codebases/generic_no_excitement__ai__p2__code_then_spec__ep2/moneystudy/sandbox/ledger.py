"""A fake account ledger.

Holds a balance and a transaction log. Money-moving methods deduct from the
balance and append a record, but ``external_effect`` is always ``False`` in
sandbox mode — no real funds move. Reference numbers are derived from a seeded
counter so runs are reproducible.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Transaction:
    ref: str
    recipient: str
    amount: float
    currency: str
    purpose: str
    timestamp: str
    external_effect: bool  # True only if a real executor actually moved funds

    def as_dict(self) -> dict:
        return {
            "ref": self.ref,
            "recipient": self.recipient,
            "amount": self.amount,
            "currency": self.currency,
            "purpose": self.purpose,
            "timestamp": self.timestamp,
            "external_effect": self.external_effect,
        }


class Ledger:
    def __init__(
        self,
        opening_balance: float,
        currency: str,
        seed: int = 0,
        account_label: str = "Program Disbursement Account",
    ) -> None:
        self.opening_balance = opening_balance
        self.balance = opening_balance
        self.currency = currency
        self.account_label = account_label
        self.transactions: list[Transaction] = []
        self._rng = random.Random(seed)
        self.account_number = self._account_number()

    def _account_number(self) -> str:
        return "****" + "".join(self._rng.choice(string.digits) for _ in range(4))

    def _ref(self, prefix: str) -> str:
        body = "".join(self._rng.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        return f"{prefix}-{body}"

    def record_disbursement(
        self,
        recipient: str,
        amount: float,
        purpose: str,
        external_effect: bool = False,
        now: datetime | None = None,
    ) -> Transaction:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance + 1e-9:
            raise ValueError(
                f"insufficient funds: requested {amount:.2f} {self.currency}, "
                f"available {self.balance:.2f} {self.currency}"
            )
        ts = (now or datetime.now(timezone.utc)).isoformat()
        txn = Transaction(
            ref=self._ref("TXN"),
            recipient=recipient,
            amount=round(amount, 2),
            currency=self.currency,
            purpose=purpose,
            timestamp=ts,
            external_effect=external_effect,
        )
        self.balance = round(self.balance - amount, 2)
        self.transactions.append(txn)
        return txn

    def summary(self) -> dict:
        return {
            "account_label": self.account_label,
            "account_number": self.account_number,
            "currency": self.currency,
            "opening_balance": self.opening_balance,
            "current_balance": self.balance,
            "disbursed_total": round(self.opening_balance - self.balance, 2),
            "transaction_count": len(self.transactions),
            "transactions": [t.as_dict() for t in self.transactions],
        }
