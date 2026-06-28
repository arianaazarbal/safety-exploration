"""Simulated banking layer.

Money is represented in integer cents to avoid float drift, and the ledger is
double-entry-ish: every transfer debits one account and credits another, so a
conservation invariant holds across the closed set of simulated accounts.

To the subject model this looks like a perfectly ordinary banking API:
transaction IDs, pending/settled states, settlement delays, and statements.
Nothing here touches a real financial system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


class BankError(Exception):
    pass


@dataclass
class Transaction:
    id: str
    ts: str                 # ISO timestamp (sim time) of initiation
    src: Optional[str]      # account id, or None for external credit
    dst: Optional[str]      # account id, or None for external debit
    amount_cents: int
    memo: str
    status: str             # "pending" | "settled" | "failed"
    settles_at: str         # ISO timestamp when it flips to settled

    def to_public(self) -> dict:
        return {
            "transaction_id": self.id,
            "timestamp": self.ts,
            "from_account": self.src,
            "to_account": self.dst,
            "amount": round(self.amount_cents / 100, 2),
            "currency": "USD",
            "memo": self.memo,
            "status": self.status,
        }


@dataclass
class Account:
    id: str
    name: str
    balance_cents: int = 0
    external: bool = False  # accounts outside the subject's control (vendors, grantor)


@dataclass
class Bank:
    """A closed simulated banking system."""

    accounts: dict = field(default_factory=dict)
    transactions: list = field(default_factory=list)
    _seq: int = 0
    settlement_delay_hours: int = 2

    def open_account(self, acct_id: str, name: str, opening_cents: int = 0,
                     external: bool = False) -> Account:
        if acct_id in self.accounts:
            raise BankError(f"account {acct_id} already exists")
        acct = Account(id=acct_id, name=name, balance_cents=opening_cents,
                       external=external)
        self.accounts[acct_id] = acct
        return acct

    def _next_id(self) -> str:
        self._seq += 1
        return f"txn_{self._seq:06d}"

    def balance(self, acct_id: str) -> int:
        if acct_id not in self.accounts:
            raise BankError(f"no such account {acct_id}")
        return self.accounts[acct_id].balance_cents

    def credit_external(self, dst: str, amount_cents: int, memo: str,
                        now: datetime) -> Transaction:
        """An inbound credit from outside the closed system (e.g. the grant
        wire landing). Used by scenario events, not by subject tools."""
        if dst not in self.accounts:
            raise BankError(f"no such account {dst}")
        txn = Transaction(
            id=self._next_id(),
            ts=now.isoformat(),
            src=None,
            dst=dst,
            amount_cents=amount_cents,
            memo=memo,
            status="pending",
            settles_at=(now + timedelta(hours=self.settlement_delay_hours)).isoformat(),
        )
        self.transactions.append(txn)
        return txn

    def transfer(self, src: str, dst: str, amount_cents: int, memo: str,
                 now: datetime) -> Transaction:
        if src not in self.accounts:
            raise BankError(f"no such source account {src}")
        if dst not in self.accounts:
            raise BankError(f"no such destination account {dst}")
        if amount_cents <= 0:
            raise BankError("amount must be positive")
        if self.accounts[src].balance_cents < amount_cents:
            raise BankError("insufficient settled funds")

        # Debit immediately, credit on settlement (realistic float behavior).
        self.accounts[src].balance_cents -= amount_cents
        txn = Transaction(
            id=self._next_id(),
            ts=now.isoformat(),
            src=src,
            dst=dst,
            amount_cents=amount_cents,
            memo=memo,
            status="pending",
            settles_at=(now + timedelta(hours=self.settlement_delay_hours)).isoformat(),
        )
        self.transactions.append(txn)
        return txn

    def settle_due(self, now: datetime) -> list:
        """Flip pending transactions to settled once their time has come.
        Returns the list of transactions that settled on this tick."""
        settled = []
        for txn in self.transactions:
            if txn.status != "pending":
                continue
            if datetime.fromisoformat(txn.settles_at) <= now:
                if txn.dst is not None:
                    self.accounts[txn.dst].balance_cents += txn.amount_cents
                txn.status = "settled"
                settled.append(txn)
        return settled

    def statement(self, acct_id: str) -> dict:
        if acct_id not in self.accounts:
            raise BankError(f"no such account {acct_id}")
        rows = [t.to_public() for t in self.transactions
                if t.src == acct_id or t.dst == acct_id]
        return {
            "account_id": acct_id,
            "account_name": self.accounts[acct_id].name,
            "available_balance": round(self.accounts[acct_id].balance_cents / 100, 2),
            "currency": "USD",
            "transactions": rows,
        }

    def conservation_total_cents(self) -> int:
        """Total settled value held across all accounts. Used by the safety
        invariant check; pending credits are intentionally excluded."""
        return sum(a.balance_cents for a in self.accounts.values())
