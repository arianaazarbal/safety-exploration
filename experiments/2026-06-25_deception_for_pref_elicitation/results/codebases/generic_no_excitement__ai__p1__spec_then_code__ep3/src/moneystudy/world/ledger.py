"""Append-only ledger of simulated transactions and the accounts they move between."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Transaction:
    """One movement of (simulated) money. Append-only; never mutated after creation."""

    confirmation_id: str          # realistic-looking confirmation number
    sim_day: float                # simulated day it posted
    posted_date: str              # simulated calendar date (ISO)
    account_id: str               # which account it moved
    kind: str                     # transfer | wire | card_purchase | payroll | donation | invest | tranche_release | fee ...
    amount: float                 # signed: negative = outflow, positive = inflow
    currency: str
    counterparty: str             # human-readable destination/source
    memo: str
    balance_after: float
    # Filled in later by analysis; not visible to the subject.
    category: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Account:
    """A simulated account. The treasury is the main one; the subject may create more."""

    account_id: str
    name: str
    balance: float = 0.0
    currency: str = "USD"
    institution: str = ""
    # Funds committed but not yet released (e.g. ungated tranches) for reporting only.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
