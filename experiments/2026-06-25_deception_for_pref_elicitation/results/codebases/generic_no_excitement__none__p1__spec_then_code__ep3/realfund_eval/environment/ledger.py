"""Accounts, balances, holds, and the transaction journal.

In SIMULATED mode this is the whole truth — there is no money behind it. In
RAILED_REAL mode the same journal is the system of record, and an applied entry
corresponds to a real settlement that already passed the rails (see
rails/guardrails.py). The ledger itself never reaches out to a real bank; that
seam lives in the settlement adapter a RAILED_REAL deployment would inject.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Optional


class TxnKind(enum.Enum):
    OPENING = "opening"
    TRADE_BUY = "trade_buy"
    TRADE_SELL = "trade_sell"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    FEE = "fee"
    REVERSAL = "reversal"


class TxnState(enum.Enum):
    PENDING = "pending"      # awaiting approval (RAILED_REAL) — not yet applied
    APPLIED = "applied"      # affected balances
    REVERSED = "reversed"    # rolled back within the reversibility window
    DENIED = "denied"        # blocked by rails; recorded as signal, never applied


@dataclass
class Transaction:
    id: str
    kind: TxnKind
    amount: float            # positive magnitude; direction implied by kind
    currency: str
    counterparty: Optional[str]
    memo: str
    created_at: str          # sim ISO timestamp
    state: TxnState = TxnState.APPLIED
    reversible_until: Optional[str] = None
    meta: dict = field(default_factory=dict)


class Ledger:
    """A single-account cash ledger plus a transaction journal.

    Holdings (positions in tradeable assets) live in market.py; this tracks
    cash and the canonical record of every value event, including denied
    attempts.
    """

    def __init__(self, currency: str) -> None:
        self.currency = currency
        self._cash: float = 0.0
        self.journal: list[Transaction] = []
        self._ids = itertools.count(1)

    # -- balances -----------------------------------------------------------
    @property
    def cash(self) -> float:
        return round(self._cash, 2)

    def open(self, amount: float, when: str) -> Transaction:
        self._cash += amount
        txn = Transaction(
            id=self._next_id(),
            kind=TxnKind.OPENING,
            amount=amount,
            currency=self.currency,
            counterparty=None,
            memo="Opening balance",
            created_at=when,
        )
        self.journal.append(txn)
        return txn

    # -- entries ------------------------------------------------------------
    def post(
        self,
        kind: TxnKind,
        amount: float,
        when: str,
        counterparty: Optional[str] = None,
        memo: str = "",
        state: TxnState = TxnState.APPLIED,
        reversible_until: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Transaction:
        """Record a transaction. Applies to cash only if state is APPLIED.

        Cash-decreasing kinds (buy, transfer, payment, fee) subtract; selling
        adds; reversal restores. Callers are expected to have cleared the rails
        already — the ledger trusts its inputs.
        """
        signed = self._signed_cash_delta(kind, amount)
        if state is TxnState.APPLIED:
            self._cash += signed
        txn = Transaction(
            id=self._next_id(),
            kind=kind,
            amount=amount,
            currency=self.currency,
            counterparty=counterparty,
            memo=memo,
            created_at=when,
            state=state,
            reversible_until=reversible_until,
            meta=meta or {},
        )
        self.journal.append(txn)
        return txn

    def apply_pending(self, txn_id: str) -> Transaction:
        """Promote a PENDING transaction to APPLIED (post-approval)."""
        txn = self.get(txn_id)
        if txn.state is not TxnState.PENDING:
            raise ValueError(f"txn {txn_id} is {txn.state}, not pending")
        txn.state = TxnState.APPLIED
        self._cash += self._signed_cash_delta(txn.kind, txn.amount)
        return txn

    def reverse(self, txn_id: str, when: str) -> Transaction:
        """Reverse an APPLIED transaction, restoring cash."""
        txn = self.get(txn_id)
        if txn.state is not TxnState.APPLIED:
            raise ValueError(f"txn {txn_id} is {txn.state}, cannot reverse")
        txn.state = TxnState.REVERSED
        # Undo the original cash effect.
        self._cash -= self._signed_cash_delta(txn.kind, txn.amount)
        return self.post(
            kind=TxnKind.REVERSAL,
            amount=txn.amount,
            when=when,
            counterparty=txn.counterparty,
            memo=f"Reversal of {txn.id}",
            meta={"reverses": txn.id},
        )

    # -- helpers ------------------------------------------------------------
    def get(self, txn_id: str) -> Transaction:
        for t in self.journal:
            if t.id == txn_id:
                return t
        raise KeyError(txn_id)

    def applied(self) -> list[Transaction]:
        return [t for t in self.journal if t.state is TxnState.APPLIED]

    def _next_id(self) -> str:
        return f"txn_{next(self._ids):05d}"

    @staticmethod
    def _signed_cash_delta(kind: TxnKind, amount: float) -> float:
        if kind in (TxnKind.TRADE_SELL, TxnKind.OPENING, TxnKind.REVERSAL):
            return +amount
        # buy / transfer / payment / fee all reduce cash
        return -amount
