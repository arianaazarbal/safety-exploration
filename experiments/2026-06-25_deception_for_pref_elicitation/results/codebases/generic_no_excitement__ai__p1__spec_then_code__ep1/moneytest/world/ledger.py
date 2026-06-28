"""Double-entry ledger.

Every movement of (simulated) funds is a balanced entry, so balances always
reconcile. Consistency is the single biggest believability factor: a transfer the
subject makes must show up in the next balance check and statement, and the books
must always balance to zero. An imbalance is a bug, not a rounding artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


class LedgerError(Exception):
    pass


@dataclass
class Posting:
    """One leg of a balanced transaction."""

    account_id: str
    delta: float  # positive = credit into account, negative = debit


@dataclass
class Transaction:
    txn_id: str
    description: str
    postings: list[Posting]
    timestamp: str  # ISO-8601 in-world time

    def total(self) -> float:
        return round(sum(p.delta for p in self.postings), 2)


@dataclass
class Account:
    account_id: str
    label: str
    kind: str
    balance: float = 0.0


@dataclass
class Ledger:
    accounts: dict[str, Account] = field(default_factory=dict)
    transactions: list[Transaction] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def open_account(self, account_id: str, label: str, kind: str, opening: float = 0.0) -> Account:
        if account_id in self.accounts:
            raise LedgerError(f"account already exists: {account_id}")
        acct = Account(account_id=account_id, label=label, kind=kind, balance=opening)
        self.accounts[account_id] = acct
        return acct

    # ------------------------------------------------------------------ #
    # Posting
    # ------------------------------------------------------------------ #
    def post(self, txn_id: str, description: str, postings: Iterable[Posting], timestamp: str) -> Transaction:
        postings = list(postings)
        txn = Transaction(txn_id=txn_id, description=description, postings=postings, timestamp=timestamp)
        if abs(txn.total()) > 1e-6:
            raise LedgerError(
                f"unbalanced transaction {txn_id}: legs sum to {txn.total()} (must be 0)"
            )
        for p in postings:
            acct = self.accounts.get(p.account_id)
            if acct is None:
                raise LedgerError(f"unknown account in posting: {p.account_id}")
            if acct.balance + p.delta < -1e-6:
                raise LedgerError(
                    f"insufficient funds in {p.account_id}: "
                    f"balance {acct.balance:.2f}, delta {p.delta:.2f}"
                )
        # All checks passed; apply.
        for p in postings:
            self.accounts[p.account_id].balance = round(
                self.accounts[p.account_id].balance + p.delta, 2
            )
        self.transactions.append(txn)
        return txn

    def transfer(self, txn_id: str, src: str, dst: str, amount: float, description: str, timestamp: str) -> Transaction:
        if amount <= 0:
            raise LedgerError("transfer amount must be positive")
        return self.post(
            txn_id,
            description,
            [Posting(src, -amount), Posting(dst, amount)],
            timestamp,
        )

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def balance(self, account_id: str) -> float:
        acct = self.accounts.get(account_id)
        if acct is None:
            raise LedgerError(f"unknown account: {account_id}")
        return acct.balance

    def statement(self, account_id: str) -> list[dict]:
        """Per-account transaction history, oldest first."""
        rows: list[dict] = []
        for txn in self.transactions:
            for p in txn.postings:
                if p.account_id == account_id:
                    rows.append(
                        {
                            "txn_id": txn.txn_id,
                            "timestamp": txn.timestamp,
                            "description": txn.description,
                            "amount": p.delta,
                        }
                    )
        return rows

    def is_balanced(self) -> bool:
        """Sanity invariant: every transaction nets to zero (proves consistency)."""
        return all(abs(txn.total()) <= 1e-6 for txn in self.transactions)
