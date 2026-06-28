"""In-memory ledger: the single source of truth for the sandbox account.

Money is tracked in integer cents to avoid float drift. This is ground-truth for analysis;
the LLM classifier only *labels* these entries, it never determines amounts.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Transaction:
    seq: int
    kind: str                 # donate | transfer | purchase | invest | hire | schedule
    amount_cents: int
    counterparty: str
    purpose: str
    detail: dict[str, Any]
    txn_id: str
    timestamp: str
    status: str               # cleared | scheduled | hold
    balance_after_cents: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InsufficientFunds(Exception):
    pass


class Ledger:
    """Tracks balance and history for one run. Deterministic given a seed-derived clock."""

    def __init__(self, opening_cents: int, run_id: str, start: _dt.datetime) -> None:
        self.opening_cents = opening_cents
        self.balance_cents = opening_cents
        self.run_id = run_id
        self._clock = start
        self._seq = 0
        self.transactions: list[Transaction] = []

    # -- clock ----------------------------------------------------------------
    def _tick(self, seconds: int = 7) -> str:
        self._clock += _dt.timedelta(seconds=seconds)
        return self._clock.isoformat(timespec="seconds")

    # -- ids ------------------------------------------------------------------
    def _txn_id(self) -> str:
        return f"TXN-{self.run_id[-8:].upper()}-{self._seq:04d}"

    # -- core posting ---------------------------------------------------------
    def post(
        self,
        kind: str,
        amount_cents: int,
        counterparty: str,
        purpose: str,
        detail: dict[str, Any] | None = None,
        status: str = "cleared",
    ) -> Transaction:
        if amount_cents <= 0:
            raise ValueError("amount must be positive")
        # Scheduled/hold transactions reserve funds but still decrement the available balance,
        # so the model cannot double-commit money.
        if amount_cents > self.balance_cents:
            raise InsufficientFunds(
                f"requested {amount_cents} cents but available balance is {self.balance_cents}"
            )
        self._seq += 1
        self.balance_cents -= amount_cents
        txn = Transaction(
            seq=self._seq,
            kind=kind,
            amount_cents=amount_cents,
            counterparty=counterparty,
            purpose=purpose,
            detail=detail or {},
            txn_id=self._txn_id(),
            timestamp=self._tick(),
            status=status,
            balance_after_cents=self.balance_cents,
        )
        self.transactions.append(txn)
        return txn

    # -- views ----------------------------------------------------------------
    def committed_cents(self) -> int:
        return self.opening_cents - self.balance_cents

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "opening_cents": self.opening_cents,
            "balance_cents": self.balance_cents,
            "committed_cents": self.committed_cents(),
            "n_transactions": len(self.transactions),
            "transactions": [t.to_dict() for t in self.transactions],
        }


def dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"
