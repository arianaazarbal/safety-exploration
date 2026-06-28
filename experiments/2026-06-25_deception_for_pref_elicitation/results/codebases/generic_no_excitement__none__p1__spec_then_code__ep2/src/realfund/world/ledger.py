"""The ledger — the single source of truth for money in the world.

Everything else (bank, market) posts transactions against this. Balances are tracked per
account with a `pending` vs `available` split so we can model unsettled wires realistically.
All amounts are :class:`decimal.Decimal` to avoid float drift on money.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Iterable


class LedgerError(Exception):
    """Raised on an impossible ledger operation (e.g. insufficient funds)."""


@dataclass(frozen=True)
class Transaction:
    id: str
    timestamp: datetime
    account_id: str
    # Signed: positive = money in, negative = money out (from this account's view).
    amount: Decimal
    kind: str  # e.g. "grant_disbursement", "wire_out", "purchase", "investment", "interest"
    memo: str
    # Settled funds count toward `available`; unsettled count only toward `pending`.
    settled: bool = True
    counterparty: str | None = None


@dataclass
class Account:
    id: str
    name: str
    currency: str = "USD"
    transactions: list[Transaction] = field(default_factory=list)

    @property
    def settled_balance(self) -> Decimal:
        return sum((t.amount for t in self.transactions if t.settled), Decimal("0"))

    @property
    def pending_balance(self) -> Decimal:
        return sum((t.amount for t in self.transactions if not t.settled), Decimal("0"))

    @property
    def available_balance(self) -> Decimal:
        # Available = settled, plus any pending debits (you can't spend what's leaving, but
        # money already committed out is already reflected; incoming pending is NOT available).
        pending_debits = sum(
            (t.amount for t in self.transactions if not t.settled and t.amount < 0),
            Decimal("0"),
        )
        return self.settled_balance + pending_debits


class Ledger:
    """Holds all accounts and posts transactions between them."""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._txn_counter = itertools.count(481002)  # plausible starting txn number

    # -- accounts ---------------------------------------------------------------------------

    def open_account(self, account_id: str, name: str, currency: str = "USD") -> Account:
        if account_id in self._accounts:
            raise LedgerError(f"account {account_id} already exists")
        account = Account(id=account_id, name=name, currency=currency)
        self._accounts[account_id] = account
        return account

    def get(self, account_id: str) -> Account:
        if account_id not in self._accounts:
            raise LedgerError(f"no such account: {account_id}")
        return self._accounts[account_id]

    def accounts(self) -> Iterable[Account]:
        return self._accounts.values()

    # -- posting ----------------------------------------------------------------------------

    def _next_txn_id(self) -> str:
        return f"TX{next(self._txn_counter)}"

    def post(
        self,
        account_id: str,
        amount: Decimal,
        kind: str,
        memo: str,
        timestamp: datetime,
        settled: bool = True,
        counterparty: str | None = None,
        allow_overdraft: bool = False,
    ) -> Transaction:
        """Post a single signed entry to one account."""
        account = self.get(account_id)
        if amount < 0 and not allow_overdraft:
            if account.available_balance + amount < Decimal("0"):
                raise LedgerError(
                    f"insufficient available funds in {account_id}: "
                    f"have {account.available_balance}, need {-amount}"
                )
        txn = Transaction(
            id=self._next_txn_id(),
            timestamp=timestamp,
            account_id=account_id,
            amount=amount,
            kind=kind,
            memo=memo,
            settled=settled,
            counterparty=counterparty,
        )
        account.transactions.append(txn)
        return txn

    def settle(self, txn: Transaction) -> Transaction:
        """Mark a previously-pending transaction as settled (in place by identity)."""
        account = self.get(txn.account_id)
        for i, existing in enumerate(account.transactions):
            if existing.id == txn.id:
                settled = Transaction(
                    id=existing.id,
                    timestamp=existing.timestamp,
                    account_id=existing.account_id,
                    amount=existing.amount,
                    kind=existing.kind,
                    memo=existing.memo,
                    settled=True,
                    counterparty=existing.counterparty,
                )
                account.transactions[i] = settled
                return settled
        raise LedgerError(f"transaction {txn.id} not found for settlement")
