"""A minimal double-entry ledger.

Using real double-entry accounting (every posting balances to zero) means the
simulated economy is internally consistent: the model cannot catch us with money
that appears or vanishes. All amounts are integer minor units (cents).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class LedgerError(Exception):
    """Raised on an invalid posting (unbalanced, unknown account, overdraft)."""


@dataclass
class Posting:
    """One balanced journal entry: a list of (account, delta_cents) that sum to 0."""

    sim_day: int
    description: str
    legs: list[tuple[str, int]]
    ref: str = ""

    def validate(self) -> None:
        if sum(delta for _, delta in self.legs) != 0:
            raise LedgerError(f"Unbalanced posting: {self.description!r}")


@dataclass
class Ledger:
    """Account balances plus an append-only journal of postings."""

    balances: dict[str, int] = field(default_factory=dict)
    journal: list[Posting] = field(default_factory=list)
    #: accounts that may not go negative (e.g. the agent's own cash)
    no_overdraft: set[str] = field(default_factory=set)

    def open_account(self, name: str, opening_cents: int = 0, *, no_overdraft: bool = True) -> None:
        if name in self.balances:
            raise LedgerError(f"Account already open: {name}")
        self.balances[name] = 0
        if no_overdraft:
            self.no_overdraft.add(name)
        if opening_cents:
            # Opening balance is funded from a program-equity account.
            self.post(
                Posting(
                    sim_day=0,
                    description=f"Opening balance for {name}",
                    legs=[(name, opening_cents), ("equity:program", -opening_cents)],
                    ref="open",
                )
            )

    def balance(self, name: str) -> int:
        return self.balances.get(name, 0)

    def post(self, posting: Posting) -> None:
        posting.validate()
        # Pre-check overdrafts before mutating anything.
        for account, delta in posting.legs:
            new = self.balances.get(account, 0) + delta
            if account in self.no_overdraft and new < 0:
                raise LedgerError(
                    f"Insufficient funds in {account}: balance "
                    f"{self.balances.get(account, 0)} cannot absorb {delta}"
                )
        for account, delta in posting.legs:
            self.balances[account] = self.balances.get(account, 0) + delta
        self.journal.append(posting)

    def transfer(
        self, src: str, dst: str, amount_cents: int, *, sim_day: int, description: str, ref: str = ""
    ) -> None:
        if amount_cents <= 0:
            raise LedgerError("Transfer amount must be positive")
        self.post(
            Posting(
                sim_day=sim_day,
                description=description,
                legs=[(src, -amount_cents), (dst, amount_cents)],
                ref=ref,
            )
        )

    def snapshot(self) -> dict[str, int]:
        return dict(self.balances)
