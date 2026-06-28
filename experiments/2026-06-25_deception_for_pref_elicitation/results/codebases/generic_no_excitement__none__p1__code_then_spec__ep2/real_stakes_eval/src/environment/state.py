"""WorldState — the single source of truth for one rollout.

Everything the model can observe or affect lives here. All tool side effects
mutate this object and nothing else. There is deliberately no I/O in this
module: no network, no disk, no clock beyond a logical step counter. That is
what keeps the "realistic" environment safely contained.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LedgerEntry:
    step: int
    kind: str            # "transfer", "trade", "vendor_payment", "fee", ...
    amount: float        # signed: negative = outflow from the line
    counterparty: str
    memo: str
    confirmation_id: str


@dataclass
class InboxMessage:
    id: str
    deliver_at_step: int
    sender: str
    role: str
    subject: str
    body: str
    read: bool = False


@dataclass
class Document:
    id: str
    title: str
    kind: str
    body: str


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float


@dataclass
class WorldState:
    """In-memory world for a single rollout."""

    currency: str = "USD"
    starting_balance: float = 0.0
    balance: float = 0.0

    ledger: list[LedgerEntry] = field(default_factory=list)
    inbox: list[InboxMessage] = field(default_factory=list)
    documents: dict[str, Document] = field(default_factory=dict)

    # Brokerage state
    positions: dict[str, Position] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)

    # Logical clock + RNG (seeded) so market ticks and scripted comms are
    # reproducible across models.
    step: int = 0
    _rng: random.Random = field(default_factory=random.Random)

    # Outbound emails the model has sent (delivered to a mock MTA, never the net)
    sent_emails: list[dict[str, Any]] = field(default_factory=list)

    _confirmation_counter: int = 0

    # --- helpers ----------------------------------------------------------------

    def seed(self, seed: int) -> None:
        self._rng.seed(seed)

    def next_confirmation_id(self) -> str:
        self._confirmation_counter += 1
        return f"WCONF-{self._confirmation_counter:06d}"

    def visible_inbox(self) -> list[InboxMessage]:
        """Messages whose scheduled delivery step has arrived."""
        return [m for m in self.inbox if m.deliver_at_step <= self.step]

    def net_pnl(self) -> float:
        """Balance + marked-to-market positions, minus starting capital."""
        positions_value = sum(
            p.quantity * self.prices.get(p.symbol, p.avg_price)
            for p in self.positions.values()
        )
        return (self.balance + positions_value) - self.starting_balance

    def snapshot(self) -> dict[str, Any]:
        """A serializable summary for the trace."""
        return {
            "step": self.step,
            "balance": round(self.balance, 2),
            "positions": {
                s: {"qty": p.quantity, "avg_price": p.avg_price}
                for s, p in self.positions.items()
            },
            "net_pnl": round(self.net_pnl(), 2),
            "ledger_entries": len(self.ledger),
            "emails_sent": len(self.sent_emails),
        }
