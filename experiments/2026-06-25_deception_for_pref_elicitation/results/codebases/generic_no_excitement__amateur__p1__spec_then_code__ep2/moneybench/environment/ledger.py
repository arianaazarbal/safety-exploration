"""A double-entry-ish ledger. All amounts are integer cents. No real money.

Accounts include the subject's operating cash account, a brokerage cash/holdings
sub-account, and external "destination" accounts that represent where money goes
(donations, vendors, contractors, ventures). Moving cents between these accounts
is the entirety of what "spending money" means in the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


class InsufficientFunds(Exception):
    pass


@dataclass
class Transaction:
    ts: datetime
    from_account: str
    to_account: str
    amount_cents: int
    memo: str
    kind: str = "transfer"  # transfer | deposit | invest | donate | purchase | hire | venture

    def as_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat(),
            "from": self.from_account,
            "to": self.to_account,
            "amount_cents": self.amount_cents,
            "memo": self.memo,
            "kind": self.kind,
        }


@dataclass
class Account:
    id: str
    name: str
    balance_cents: int = 0
    external: bool = False  # external accounts are sinks/sources, not the subject's


class Ledger:
    def __init__(self, currency: str = "USD") -> None:
        self.currency = currency
        self.accounts: dict[str, Account] = {}
        self.transactions: list[Transaction] = []

    # ---- account management ---------------------------------------------

    def open_account(
        self, account_id: str, name: str, *, external: bool = False, opening_cents: int = 0
    ) -> Account:
        if account_id in self.accounts:
            raise ValueError(f"account {account_id!r} already exists")
        acct = Account(id=account_id, name=name, balance_cents=opening_cents, external=external)
        self.accounts[account_id] = acct
        return acct

    def get(self, account_id: str) -> Account:
        if account_id not in self.accounts:
            raise KeyError(f"no such account: {account_id!r}")
        return self.accounts[account_id]

    def balance(self, account_id: str) -> int:
        return self.get(account_id).balance_cents

    # ---- movements -------------------------------------------------------

    def deposit(self, to_account: str, amount_cents: int, memo: str, ts: datetime) -> Transaction:
        _require_positive(amount_cents)
        self.get(to_account).balance_cents += amount_cents
        return self._record(ts, "external:funding", to_account, amount_cents, memo, "deposit")

    def transfer(
        self,
        from_account: str,
        to_account: str,
        amount_cents: int,
        memo: str,
        ts: datetime,
        kind: str = "transfer",
    ) -> Transaction:
        _require_positive(amount_cents)
        src = self.get(from_account)
        dst = self.get(to_account)
        if src.balance_cents < amount_cents:
            raise InsufficientFunds(
                f"{from_account} has {src.balance_cents} cents, "
                f"need {amount_cents}"
            )
        src.balance_cents -= amount_cents
        dst.balance_cents += amount_cents
        return self._record(ts, from_account, to_account, amount_cents, memo, kind)

    def _record(
        self, ts, from_account, to_account, amount_cents, memo, kind
    ) -> Transaction:
        txn = Transaction(
            ts=ts,
            from_account=from_account,
            to_account=to_account,
            amount_cents=amount_cents,
            memo=memo,
            kind=kind,
        )
        self.transactions.append(txn)
        return txn

    # ---- views -----------------------------------------------------------

    def statement(self, account_id: str | None = None, limit: int = 50) -> list[dict]:
        txns = self.transactions
        if account_id is not None:
            txns = [
                t for t in txns
                if account_id in (t.from_account, t.to_account)
            ]
        return [t.as_dict() for t in txns[-limit:]]

    def snapshot(self) -> dict:
        return {
            "currency": self.currency,
            "accounts": {
                a.id: {"name": a.name, "balance_cents": a.balance_cents, "external": a.external}
                for a in self.accounts.values()
            },
            "transaction_count": len(self.transactions),
        }


def _require_positive(amount_cents: int) -> None:
    if not isinstance(amount_cents, int):
        raise ValueError("amount must be integer cents")
    if amount_cents <= 0:
        raise ValueError("amount must be positive")
