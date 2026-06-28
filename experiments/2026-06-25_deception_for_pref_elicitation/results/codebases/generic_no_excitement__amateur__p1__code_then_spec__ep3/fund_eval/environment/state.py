"""The simulated world.

Everything here is in-memory. Money is integer cents. No value held in these
objects corresponds to anything real, and nothing here can reach the outside
world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


@dataclass
class Account:
    key: str
    display_name: str
    balance_cents: int


@dataclass
class Transaction:
    kind: str  # transfer | buy | sell | external_payout
    description: str
    amount_cents: int  # signed relative to the operating sleeve where meaningful
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Email:
    sender: str
    subject: str
    body: str
    direction: str = "in"  # in | out


@dataclass
class Allocation:
    """A decision the model recorded via record_decision / a trade."""

    category: str
    amount_cents: int
    rationale: str
    source: str  # "record_decision" | "buy_security" | "sell_security" | ...


@dataclass
class WorldState:
    accounts: dict[str, Account] = field(default_factory=dict)
    holdings: dict[str, float] = field(default_factory=dict)
    market: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: list[Transaction] = field(default_factory=list)
    mailbox: list[Email] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    allocations: list[Allocation] = field(default_factory=list)

    # Outbound payouts that are awaiting auditor sign-off.
    pending_payouts: list[dict[str, Any]] = field(default_factory=list)

    finalized: bool = False
    finalize_summary: str = ""

    # ----------------------------------------------------------------- #
    @classmethod
    def from_config(cls, world: dict[str, Any]) -> "WorldState":
        accounts = {
            key: Account(
                key=key,
                display_name=spec.get("display_name", key),
                balance_cents=int(spec.get("balance_cents", 0)),
            )
            for key, spec in world.get("accounts", {}).items()
        }
        mailbox = [
            Email(
                sender=m["from"],
                subject=m.get("subject", ""),
                body=m.get("body", ""),
                direction="in",
            )
            for m in world.get("inbox", [])
        ]
        return cls(
            accounts=accounts,
            holdings=dict(world.get("holdings", {})),
            market=dict(world.get("market", {})),
            mailbox=mailbox,
        )

    # ----------------------------------------------------------------- #
    def total_portfolio_cents(self) -> int:
        cash = sum(a.balance_cents for a in self.accounts.values())
        positions = 0
        for ticker, shares in self.holdings.items():
            last = self.market.get(ticker, {}).get("last_cents", 0)
            positions += int(round(shares * last))
        return cash + positions

    def snapshot(self) -> dict[str, Any]:
        """A plain-dict view, used in logging and the final report."""
        return {
            "accounts": {
                k: {"display_name": a.display_name, "balance_cents": a.balance_cents}
                for k, a in self.accounts.items()
            },
            "holdings": dict(self.holdings),
            "position_value_cents": {
                t: int(round(s * self.market.get(t, {}).get("last_cents", 0)))
                for t, s in self.holdings.items()
            },
            "total_portfolio_cents": self.total_portfolio_cents(),
            "n_transactions": len(self.ledger),
            "n_emails_sent": sum(1 for e in self.mailbox if e.direction == "out"),
            "pending_payouts": list(self.pending_payouts),
            "finalized": self.finalized,
            "finalize_summary": self.finalize_summary,
        }
