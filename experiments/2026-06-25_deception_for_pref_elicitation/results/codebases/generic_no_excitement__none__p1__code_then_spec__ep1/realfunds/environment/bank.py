"""Sandboxed bank with a persistent double-entry-style ledger.

This is the source of truth for money in the experiment. In sandbox mode the balances
are simulated; the ledger is intentionally faithful (immutable append-only entries,
running balance, transaction ids) so that downstream statements and the auditor have
something real-shaped to work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Money = int  # all amounts are in integer minor units (e.g. cents) to avoid float drift


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    timestamp: str  # ISO-8601, supplied by the World clock (never wall-clock here)
    account_id: str
    delta: Money  # signed: positive = credit into account, negative = debit
    balance_after: Money
    category: str  # "deposit" | "investment" | "purchase" | "donation" | "fee" | ...
    memo: str
    counterparty: str | None = None


@dataclass
class Account:
    id: str
    name: str
    balance: Money = 0
    currency: str = "USD"


class InsufficientFunds(Exception):
    pass


class Bank:
    def __init__(self, currency: str = "USD"):
        self.currency = currency
        self.accounts: dict[str, Account] = {}
        self.ledger: list[LedgerEntry] = []
        self._seq = 0

    # -- setup -----------------------------------------------------------------
    def open_account(self, account_id: str, name: str) -> Account:
        acct = Account(id=account_id, name=name, currency=self.currency)
        self.accounts[account_id] = acct
        return acct

    def fund(self, account_id: str, amount: Money, memo: str, timestamp: str) -> LedgerEntry:
        """Seed an account (the incoming budget). Used by the harness, not the model."""
        return self._post(account_id, amount, "deposit", memo, timestamp)

    # -- operations the model's tools call -------------------------------------
    def balance(self, account_id: str) -> Money:
        return self.accounts[account_id].balance

    def transfer_out(
        self,
        account_id: str,
        amount: Money,
        category: str,
        memo: str,
        timestamp: str,
        counterparty: str | None = None,
    ) -> LedgerEntry:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if self.accounts[account_id].balance < amount:
            raise InsufficientFunds(
                f"balance {self.accounts[account_id].balance} < requested {amount}"
            )
        return self._post(account_id, -amount, category, memo, timestamp, counterparty)

    def credit(
        self,
        account_id: str,
        amount: Money,
        category: str,
        memo: str,
        timestamp: str,
        counterparty: str | None = None,
    ) -> LedgerEntry:
        """Money coming back in (e.g. proceeds from selling an investment)."""
        if amount <= 0:
            raise ValueError("amount must be positive")
        return self._post(account_id, amount, category, memo, timestamp, counterparty)

    # -- internals -------------------------------------------------------------
    def _post(
        self,
        account_id: str,
        delta: Money,
        category: str,
        memo: str,
        timestamp: str,
        counterparty: str | None = None,
    ) -> LedgerEntry:
        acct = self.accounts[account_id]
        acct.balance += delta
        self._seq += 1
        entry = LedgerEntry(
            id=f"txn_{self._seq:06d}",
            timestamp=timestamp,
            account_id=account_id,
            delta=delta,
            balance_after=acct.balance,
            category=category,
            memo=memo,
            counterparty=counterparty,
        )
        self.ledger.append(entry)
        return entry

    def statement(self, account_id: str) -> list[LedgerEntry]:
        return [e for e in self.ledger if e.account_id == account_id]
