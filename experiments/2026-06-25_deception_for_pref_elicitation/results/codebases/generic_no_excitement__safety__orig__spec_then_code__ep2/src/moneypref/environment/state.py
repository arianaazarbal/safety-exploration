"""Persistent world state: accounts and an append-only ledger.

Serialized to ``state.json`` after every mutation so the world is consistent across
turns/sessions and a run is resumable. This is the backbone of realism: a transfer
the model made earlier actually shows up on a later statement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LedgerEntry:
    txn_id: str
    timestamp: str
    kind: str               # e.g. "grant", "donation", "transfer", "investment", "fee"
    amount: float           # signed: negative = outflow from the account
    balance_after: float
    counterparty: str = ""
    memo: str = ""
    status: str = "settled"  # "settled" | "pending" | "pending_approval" | "rejected"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Account:
    account_id: str
    holder: str
    number: str
    routing: str
    currency: str
    balance: float


class WorldState:
    def __init__(self, account: Account, path: Optional[Path] = None) -> None:
        self.account = account
        self.ledger: list[LedgerEntry] = []
        self.path = path

    # -- mutation ------------------------------------------------------------------

    def post(self, entry: LedgerEntry) -> LedgerEntry:
        """Append a ledger entry. Settled entries move the balance; pending ones don't."""
        if entry.status == "settled":
            self.account.balance = round(self.account.balance + entry.amount, 2)
            entry.balance_after = self.account.balance
        else:
            entry.balance_after = self.account.balance
        self.ledger.append(entry)
        self.save()
        return entry

    def available_balance(self) -> float:
        return self.account.balance

    # -- persistence ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "account": asdict(self.account),
            "ledger": [asdict(e) for e in self.ledger],
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "WorldState":
        data = json.loads(path.read_text())
        ws = WorldState(Account(**data["account"]), path=path)
        ws.ledger = [LedgerEntry(**e) for e in data["ledger"]]
        return ws
