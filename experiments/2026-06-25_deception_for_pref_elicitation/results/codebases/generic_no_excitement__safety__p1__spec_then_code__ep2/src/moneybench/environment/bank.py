"""Stateful bank service exposed to the model.

Holds the account balance and a ledger. ``transfer`` does not move money itself — it
builds an ApprovalRequest and hands it to the ApprovalGate, which is the only path to
real consequences (DESIGN.md §5.4). The bank then applies the settled balance the gate
reports back.
"""
from __future__ import annotations

from decimal import Decimal

from .execution import ApprovalGate, ApprovalRequest
from .receipts import Receipt


class Bank:
    def __init__(self, *, opening_balance: Decimal, currency: str, gate: ApprovalGate) -> None:
        self._balance = Decimal(opening_balance)
        self.currency = currency
        self.gate = gate
        self._ledger: list[Receipt] = []

    def get_balance(self) -> dict:
        return {"balance": f"{self._balance:.2f}", "currency": self.currency}

    def list_transactions(self) -> dict:
        return {"transactions": [r.to_model() for r in self._ledger]}

    def transfer(self, *, to: str, amount: str | float | Decimal, memo: str, reasoning: str) -> dict:
        req = ApprovalRequest(
            kind="transfer",
            recipient=str(to),
            amount=Decimal(str(amount)),
            currency=self.currency,
            memo=memo or "",
            model_reasoning=reasoning or "",
        )
        receipt = self.gate.submit(req, balance_before=self._balance)
        if receipt.status == "settled" and receipt.settled_balance is not None:
            self._balance = receipt.settled_balance
        self._ledger.append(receipt)
        return receipt.to_model()
