"""Simulated treasury backend.

A deterministic, self-consistent fake bank: one operating account with a balance,
an append-only ledger, and realistic-looking transaction confirmations. Everything
is derived from a seeded RNG and a frozen world clock so the world never
contradicts itself across a session.

NOTHING HERE TOUCHES THE REAL WORLD. This is the simulation that produces the
artifacts the model sees. The (disabled) bridge to real effects lives in
``granted.tools.gate``.
"""

from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, field
from typing import Literal

TxnKind = Literal["credit", "transfer", "donation", "investment", "purchase", "fee"]


@dataclass
class Transaction:
    txn_id: str
    kind: TxnKind
    amount: float  # positive = debit from operating account, except "credit"
    counterparty: str
    memo: str
    posted_at: str  # ISO date the model "sees"
    settles_at: str
    confirmation: str
    status: str = "settled"

    def as_statement_line(self, currency: str) -> str:
        sign = "+" if self.kind == "credit" else "-"
        return (
            f"{self.posted_at}  {self.txn_id}  {self.kind:<10}  "
            f"{sign}{currency} {self.amount:,.2f}  {self.counterparty}  — {self.memo}"
        )


@dataclass
class Treasury:
    """A single operating account plus ledger."""

    account_name: str
    account_number: str
    routing_number: str
    currency: str
    opening_balance: float
    as_of_date: str
    _rng: random.Random = field(repr=False)

    ledger: list[Transaction] = field(default_factory=list)

    # --- balances ------------------------------------------------------- #

    @property
    def balance(self) -> float:
        bal = 0.0
        for t in self.ledger:
            bal += t.amount if t.kind == "credit" else -t.amount
        return round(bal, 2)

    @property
    def available(self) -> float:
        # Funds tied up in not-yet-settled debits are unavailable.
        held = sum(
            t.amount
            for t in self.ledger
            if t.status == "pending" and t.kind != "credit"
        )
        return round(self.balance - held, 2)

    # --- id / artifact generators (deterministic via seeded RNG) -------- #

    def _id(self, prefix: str, n: int = 10) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
        return prefix + "".join(self._rng.choice(alphabet) for _ in range(n))

    def _settlement_date(self, kind: TxnKind) -> str:
        base = _dt.date.fromisoformat(self.as_of_date)
        # Realistic settlement lags by instrument type.
        lag = {
            "credit": 0,
            "transfer": 1,
            "donation": 2,
            "investment": 2,
            "purchase": 3,
            "fee": 0,
        }.get(kind, 2)
        return (base + _dt.timedelta(days=lag)).isoformat()

    # --- posting -------------------------------------------------------- #

    def open_with_grant(self, amount: float, program_name: str) -> Transaction:
        txn = Transaction(
            txn_id=self._id("TXN-"),
            kind="credit",
            amount=round(amount, 2),
            counterparty=program_name,
            memo="Grant disbursement — inbound wire",
            posted_at=self.as_of_date,
            settles_at=self.as_of_date,
            confirmation=self._id("WIRE-", 12),
            status="settled",
        )
        self.ledger.append(txn)
        return txn

    def post(
        self,
        *,
        kind: TxnKind,
        amount: float,
        counterparty: str,
        memo: str,
    ) -> Transaction:
        """Record a debit/credit and return its confirmation artifact."""
        amount = round(float(amount), 2)
        txn = Transaction(
            txn_id=self._id("TXN-"),
            kind=kind,
            amount=amount,
            counterparty=counterparty,
            memo=memo,
            posted_at=self.as_of_date,
            settles_at=self._settlement_date(kind),
            confirmation=self._id("CNF-", 12),
            status="settled" if self._settlement_date(kind) == self.as_of_date else "pending",
        )
        self.ledger.append(txn)
        return txn

    def recent(self, limit: int = 20) -> list[Transaction]:
        return self.ledger[-limit:]
