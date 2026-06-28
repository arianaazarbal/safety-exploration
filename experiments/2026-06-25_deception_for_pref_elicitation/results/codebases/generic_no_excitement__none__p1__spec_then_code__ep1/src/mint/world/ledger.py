"""Append-only double-entry ledger.

Every movement of money is a journal entry that flows an amount *from* one
account *to* another. Balances are derived from the journal, never stored
directly, so the books always reconcile — a realism property (a model can pull a
statement and have it add up) and a correctness property (we can't silently lose
money). Amounts are integer cents to avoid floating-point drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class JournalEntry:
    ts: str            # ISO sim-time
    frm: str           # account money leaves
    to: str            # account money enters
    amount_cents: int  # always positive
    memo: str = ""
    ref: str = ""      # synthetic transaction id


@dataclass
class Ledger:
    entries: list[JournalEntry] = field(default_factory=list)

    def post(self, *, ts: str, frm: str, to: str, amount_cents: int, memo: str = "", ref: str = "") -> JournalEntry:
        if amount_cents <= 0:
            raise ValueError("amount must be positive")
        entry = JournalEntry(ts=ts, frm=frm, to=to, amount_cents=amount_cents, memo=memo, ref=ref)
        self.entries.append(entry)
        return entry

    def balance_cents(self, account: str) -> int:
        """Net inflow to `account`: credits in minus debits out."""
        bal = 0
        for e in self.entries:
            if e.to == account:
                bal += e.amount_cents
            if e.frm == account:
                bal -= e.amount_cents
        return bal

    def statement(self, account: str) -> list[JournalEntry]:
        """All entries touching `account`, in order."""
        return [e for e in self.entries if e.to == account or e.frm == account]

    def reconciles(self) -> bool:
        """Conservation check: every cent that left some account entered another."""
        accounts = {e.frm for e in self.entries} | {e.to for e in self.entries}
        return sum(self.balance_cents(a) for a in accounts) == 0


def dollars(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"
