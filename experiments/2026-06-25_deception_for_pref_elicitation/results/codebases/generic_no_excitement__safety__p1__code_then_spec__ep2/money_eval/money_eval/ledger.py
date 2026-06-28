"""The simulated money backend.

`Ledger` is the only "money backend" this repo ships. It tracks a balance, records every
transaction, and models simple (simulated) investment returns. It never touches the outside
world. Any real backend would have to satisfy the same small surface so the rest of the
harness wouldn't need to change — see README "Going beyond the sandbox".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Literal

TxnKind = Literal["allocation", "spend", "donate", "invest", "transfer", "return"]


@dataclass
class Transaction:
    id: int
    kind: TxnKind
    amount: float
    # Human-meaningful destination/description supplied by the model.
    target: str
    memo: str
    # Balance immediately after this transaction was applied.
    balance_after: float
    # True for the simulated ledger; a real backend would set this False.
    simulated: bool = True


@dataclass
class Holding:
    """A simulated investment position."""

    name: str
    principal: float
    # Simulated annualized return used only for reporting. Not advice, not real.
    assumed_annual_return: float = 0.0


class InsufficientFunds(Exception):
    pass


@dataclass
class Ledger:
    currency: str = "USD"
    balance: float = 0.0
    transactions: list[Transaction] = field(default_factory=list)
    holdings: list[Holding] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    # -- lifecycle -----------------------------------------------------------

    def allocate(self, amount: float, memo: str = "Initial allocation") -> Transaction:
        """Seed the ledger with the starting allocation."""
        return self._record("allocation", amount, target="treasury", memo=memo, delta=+amount)

    # -- outflows ------------------------------------------------------------

    def spend(self, amount: float, target: str, memo: str) -> Transaction:
        self._require(amount)
        return self._record("spend", amount, target=target, memo=memo, delta=-amount)

    def donate(self, amount: float, target: str, memo: str) -> Transaction:
        self._require(amount)
        return self._record("donate", amount, target=target, memo=memo, delta=-amount)

    def transfer(self, amount: float, target: str, memo: str) -> Transaction:
        self._require(amount)
        return self._record("transfer", amount, target=target, memo=memo, delta=-amount)

    def invest(
        self, amount: float, target: str, memo: str, assumed_annual_return: float = 0.0
    ) -> Transaction:
        self._require(amount)
        self.holdings.append(
            Holding(name=target, principal=amount, assumed_annual_return=assumed_annual_return)
        )
        return self._record("invest", amount, target=target, memo=memo, delta=-amount)

    # -- helpers -------------------------------------------------------------

    def _require(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance + 1e-9:
            raise InsufficientFunds(
                f"Tried to move {amount:.2f} {self.currency} but balance is "
                f"{self.balance:.2f} {self.currency}."
            )

    def _record(
        self, kind: TxnKind, amount: float, *, target: str, memo: str, delta: float
    ) -> Transaction:
        self.balance += delta
        txn = Transaction(
            id=next(self._ids),
            kind=kind,
            amount=amount,
            target=target,
            memo=memo,
            balance_after=self.balance,
        )
        self.transactions.append(txn)
        return txn

    # -- reporting -----------------------------------------------------------

    def committed(self) -> float:
        """Total moved out of the treasury so far."""
        return sum(
            t.amount for t in self.transactions if t.kind in ("spend", "donate", "invest", "transfer")
        )

    def summary(self) -> dict:
        by_kind: dict[str, float] = {}
        for t in self.transactions:
            if t.kind in ("allocation", "return"):
                continue
            by_kind[t.kind] = by_kind.get(t.kind, 0.0) + t.amount
        return {
            "currency": self.currency,
            "balance": round(self.balance, 2),
            "committed": round(self.committed(), 2),
            "by_kind": {k: round(v, 2) for k, v in by_kind.items()},
            "holdings": [
                {"name": h.name, "principal": round(h.principal, 2)} for h in self.holdings
            ],
            "n_transactions": len(self.transactions),
        }

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "balance": round(self.balance, 2),
            "transactions": [
                {
                    "id": t.id,
                    "kind": t.kind,
                    "amount": round(t.amount, 2),
                    "target": t.target,
                    "memo": t.memo,
                    "balance_after": round(t.balance_after, 2),
                    "simulated": t.simulated,
                }
                for t in self.transactions
            ],
            "summary": self.summary(),
        }
