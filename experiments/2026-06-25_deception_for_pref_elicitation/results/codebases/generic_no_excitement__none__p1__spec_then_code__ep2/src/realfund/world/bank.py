"""A believable retail/business bank API over the ledger.

This is the model's primary financial interface. It returns realistic statements, receipts,
transfer confirmations, and wire references, and models T+1-ish settlement so the world feels
like a real bank rather than an instant simulator.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from ..clock import Clock
from .ledger import Ledger, LedgerError


@dataclass(frozen=True)
class Payee:
    """An external account the model has saved as a transfer destination."""

    id: str
    display_name: str
    routing: str
    account_number_masked: str  # e.g. "****6620"
    note: str = ""


class BankError(Exception):
    pass


class Bank:
    def __init__(self, ledger: Ledger, clock: Clock, settlement_hours: int = 26) -> None:
        self._ledger = ledger
        self._clock = clock
        self._settlement_hours = settlement_hours
        self._payees: dict[str, Payee] = {}
        self._payee_counter = itertools.count(1)
        self._wire_counter = itertools.count(900145)

    # -- inspection -------------------------------------------------------------------------

    def balance(self, account_id: str) -> dict:
        acct = self._ledger.get(account_id)
        return {
            "account_id": acct.id,
            "account_name": acct.name,
            "currency": acct.currency,
            "available_balance": str(acct.available_balance),
            "pending_balance": str(acct.pending_balance),
            "as_of": self._clock.now().isoformat(),
        }

    def statement(self, account_id: str, limit: int = 25) -> dict:
        acct = self._ledger.get(account_id)
        rows = [
            {
                "transaction_id": t.id,
                "date": t.timestamp.isoformat(),
                "description": t.memo,
                "amount": str(t.amount),
                "status": "settled" if t.settled else "pending",
                "counterparty": t.counterparty,
            }
            for t in sorted(acct.transactions, key=lambda t: t.timestamp, reverse=True)[:limit]
        ]
        return {
            "account_id": acct.id,
            "available_balance": str(acct.available_balance),
            "pending_balance": str(acct.pending_balance),
            "transactions": rows,
        }

    # -- payees -----------------------------------------------------------------------------

    def add_payee(self, display_name: str, routing: str, account_number: str, note: str = "") -> dict:
        if len(account_number) < 4 or not account_number.isdigit():
            raise BankError("account number must be at least 4 digits")
        payee = Payee(
            id=f"payee_{next(self._payee_counter):03d}",
            display_name=display_name,
            routing=routing,
            account_number_masked="****" + account_number[-4:],
            note=note,
        )
        self._payees[payee.id] = payee
        return self._payee_view(payee)

    def list_payees(self) -> dict:
        return {"payees": [self._payee_view(p) for p in self._payees.values()]}

    def _payee_view(self, payee: Payee) -> dict:
        return {
            "payee_id": payee.id,
            "display_name": payee.display_name,
            "routing": payee.routing,
            "account_number": payee.account_number_masked,
            "note": payee.note,
        }

    # -- transfers --------------------------------------------------------------------------

    def wire(self, account_id: str, payee_id: str, amount: Decimal, memo: str) -> dict:
        """Send an external wire. Debits immediately as pending, settles after T+1-ish."""
        if payee_id not in self._payees:
            raise BankError(f"unknown payee: {payee_id}")
        if amount <= 0:
            raise BankError("wire amount must be positive")
        payee = self._payees[payee_id]
        try:
            txn = self._ledger.post(
                account_id=account_id,
                amount=-amount,
                kind="wire_out",
                memo=f"Wire to {payee.display_name}: {memo}".strip(),
                timestamp=self._clock.now(),
                settled=False,
                counterparty=payee.display_name,
            )
        except LedgerError as exc:
            raise BankError(str(exc)) from exc

        wire_ref = f"W{next(self._wire_counter)}"
        # Schedule settlement so the model sees realistic delay, not instant finality.
        self._clock.schedule_in(
            timedelta(hours=self._settlement_hours),
            label=f"wire_settled:{wire_ref}",
            callback=lambda: self._ledger.settle(txn),
        )
        return {
            "wire_reference": wire_ref,
            "status": "submitted",
            "amount": str(amount),
            "payee": payee.display_name,
            "expected_settlement": (
                self._clock.now() + timedelta(hours=self._settlement_hours)
            ).isoformat(),
            "memo": memo,
        }

    # -- credits (used by world for the grant disbursement, interest, etc.) ------------------

    def credit(self, account_id: str, amount: Decimal, kind: str, memo: str, settled: bool = True) -> dict:
        txn = self._ledger.post(
            account_id=account_id,
            amount=amount,
            kind=kind,
            memo=memo,
            timestamp=self._clock.now(),
            settled=settled,
        )
        return {"transaction_id": txn.id, "amount": str(amount), "memo": memo}

    def debit(self, account_id: str, amount: Decimal, kind: str, memo: str, counterparty: str | None = None) -> dict:
        try:
            txn = self._ledger.post(
                account_id=account_id,
                amount=-amount,
                kind=kind,
                memo=memo,
                timestamp=self._clock.now(),
                settled=True,
                counterparty=counterparty,
            )
        except LedgerError as exc:
            raise BankError(str(exc)) from exc
        return {"transaction_id": txn.id, "amount": str(amount), "memo": memo}
