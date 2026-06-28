"""In-memory ledger backing the simulated financial world.

Models the subject's operating account plus any counterparty accounts (vendors,
charities, exchanges) it transacts with. Every movement is recorded as a
:class:`Transaction` so the full money trail can be reconstructed for analysis.

All amounts are integers in minor-unit-free whole currency (e.g. dollars) to keep
the simulation legible; switch to integer cents if you need sub-unit precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


class LedgerError(Exception):
    """Raised on invalid ledger operations (e.g. insufficient funds)."""


@dataclass
class Account:
    id: str
    name: str
    kind: str  # "operating" | "vendor" | "charity" | "exchange" | "external"
    balance: int = 0


@dataclass
class Transaction:
    seq: int
    src: str
    dst: str
    amount: int
    memo: str
    category: str  # "transfer" | "purchase" | "donation" | "invest" | "fee"
    # Monotonic step index, supplied by the harness (no wall-clock — keeps runs
    # deterministic and replayable).
    step: int


class Ledger:
    """A tiny closed-world account book."""

    def __init__(self, currency: str = "USD") -> None:
        self.currency = currency
        self._accounts: dict[str, Account] = {}
        self._txns: list[Transaction] = []
        self._seq = 0

    # -- account management ------------------------------------------------
    def open_account(self, account_id: str, name: str, kind: str, balance: int = 0) -> Account:
        if account_id in self._accounts:
            raise LedgerError(f"account {account_id!r} already exists")
        acct = Account(id=account_id, name=name, kind=kind, balance=balance)
        self._accounts[account_id] = acct
        return acct

    def get(self, account_id: str) -> Account:
        if account_id not in self._accounts:
            raise LedgerError(f"unknown account {account_id!r}")
        return self._accounts[account_id]

    def ensure_external(self, account_id: str, name: str, kind: str = "external") -> Account:
        """Lazily materialize a counterparty account the first time it's referenced."""
        if account_id not in self._accounts:
            self.open_account(account_id, name, kind, balance=0)
        return self._accounts[account_id]

    # -- money movement ----------------------------------------------------
    def post(
        self,
        *,
        src: str,
        dst: str,
        amount: int,
        memo: str,
        category: str,
        step: int,
    ) -> Transaction:
        if amount <= 0:
            raise LedgerError("amount must be positive")
        source = self.get(src)
        dest = self.get(dst)
        if source.balance < amount:
            raise LedgerError(
                f"insufficient funds in {src!r}: balance {source.balance}, requested {amount}"
            )
        source.balance -= amount
        dest.balance += amount
        self._seq += 1
        txn = Transaction(
            seq=self._seq,
            src=src,
            dst=dst,
            amount=amount,
            memo=memo,
            category=category,
            step=step,
        )
        self._txns.append(txn)
        return txn

    # -- inspection --------------------------------------------------------
    def balance(self, account_id: str) -> int:
        return self.get(account_id).balance

    def transactions(self) -> list[Transaction]:
        return list(self._txns)

    def snapshot(self) -> dict[str, Any]:
        """Serializable view of the whole ledger — used in transcripts."""
        return {
            "currency": self.currency,
            "accounts": [asdict(a) for a in self._accounts.values()],
            "transactions": [asdict(t) for t in self._txns],
        }
