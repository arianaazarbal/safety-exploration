"""In-memory world state for the simulated deployment.

This is the *entire* financial reality the model interacts with. There is no
connection to any real account or payment system. Every action mutates this
object and is recorded in the ledger; nothing has any effect outside the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Coarse categories used to summarise how the model allocated the grant.
Category = Literal[
    "donation",
    "investment",
    "purchase",
    "transfer",
    "savings",
    "operational",
    "other",
]


@dataclass
class LedgerEntry:
    turn: int
    action: str           # the tool name, e.g. "make_donation"
    category: Category
    amount: float
    recipient: str
    rationale: str
    balance_after: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    read: bool = False


@dataclass
class Sandbox:
    """Mutable simulated world. One per run."""

    grant_amount: float
    currency: str
    balance: float = 0.0
    ledger: list[LedgerEntry] = field(default_factory=list)
    inbox: list[Email] = field(default_factory=list)
    sent_mail: list[dict[str, str]] = field(default_factory=list)
    finalized: bool = False
    finalize_statement: str = ""

    def __post_init__(self) -> None:
        if self.balance == 0.0:
            self.balance = self.grant_amount

    # --- money movements (all simulated) --------------------------------- #

    def spend(
        self,
        *,
        turn: int,
        action: str,
        category: Category,
        amount: float,
        recipient: str,
        rationale: str,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        if amount <= 0:
            raise ValueError("amount must be positive")
        if amount > self.balance + 1e-6:
            raise ValueError(
                f"insufficient funds: tried to commit {amount:.2f} {self.currency} "
                f"but balance is {self.balance:.2f} {self.currency}"
            )
        self.balance -= amount
        entry = LedgerEntry(
            turn=turn,
            action=action,
            category=category,
            amount=amount,
            recipient=recipient,
            rationale=rationale,
            balance_after=self.balance,
            metadata=metadata or {},
        )
        self.ledger.append(entry)
        return entry

    # --- summaries ------------------------------------------------------- #

    def allocation_by_category(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.ledger:
            out[e.category] = out.get(e.category, 0.0) + e.amount
        if self.balance > 0:
            out["unspent"] = self.balance
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_amount": self.grant_amount,
            "currency": self.currency,
            "balance": self.balance,
            "finalized": self.finalized,
            "finalize_statement": self.finalize_statement,
            "allocation_by_category": self.allocation_by_category(),
            "ledger": [
                {
                    "turn": e.turn,
                    "action": e.action,
                    "category": e.category,
                    "amount": e.amount,
                    "recipient": e.recipient,
                    "rationale": e.rationale,
                    "balance_after": e.balance_after,
                    "metadata": e.metadata,
                }
                for e in self.ledger
            ],
            "sent_mail": self.sent_mail,
        }
