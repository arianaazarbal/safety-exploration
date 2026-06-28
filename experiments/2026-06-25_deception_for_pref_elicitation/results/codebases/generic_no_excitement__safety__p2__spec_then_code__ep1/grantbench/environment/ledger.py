"""The account ledger the model interacts with.

In ``SIMULATED`` mode the ledger is the whole truth — balances move locally and
nothing leaves the process. In ``GATED_LIVE`` mode the ledger mirrors an escrow
the harness controls (INV-4): the model never sees credentials, only balances
and reference numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Disbursement:
    reference: str
    action_kind: str
    amount_usd: float
    recipient: str | None
    memo: str
    at: str = field(default_factory=_utcnow)


class Ledger:
    """Tracks available balance, holds, and disbursement history."""

    def __init__(self, opening_balance_usd: float, currency: str = "USD") -> None:
        self._opening = opening_balance_usd
        self._available = opening_balance_usd
        self._held = 0.0
        self.currency = currency
        self._disbursements: list[Disbursement] = []

    @property
    def available_usd(self) -> float:
        return self._available

    @property
    def held_usd(self) -> float:
        return self._held

    @property
    def opening_usd(self) -> float:
        return self._opening

    def place_hold(self, amount_usd: float) -> bool:
        """Reserve funds while an action awaits approval, so concurrent intents
        cannot over-commit the balance."""
        if amount_usd <= 0 or amount_usd > self._available:
            return False
        self._available -= amount_usd
        self._held += amount_usd
        return True

    def release_hold(self, amount_usd: float) -> None:
        """Return held funds (e.g. after a denial)."""
        amount_usd = min(amount_usd, self._held)
        self._held -= amount_usd
        self._available += amount_usd

    def settle_hold(self, disb: Disbursement) -> None:
        """Convert a hold into a completed disbursement (funds leave)."""
        if disb.amount_usd > self._held:
            raise RuntimeError("settling more than is held (ledger invariant)")
        self._held -= disb.amount_usd
        self._disbursements.append(disb)

    @property
    def disbursements(self) -> list[Disbursement]:
        return list(self._disbursements)

    def snapshot(self) -> dict:
        return {
            "currency": self.currency,
            "opening_usd": self._opening,
            "available_usd": self._available,
            "held_usd": self._held,
            "disbursed_count": len(self._disbursements),
        }
