"""Ties the simulated backend, a frozen world clock, and the organizational context
into one stateful world. The single source of truth for an episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..config import GrantConfig
from . import artifacts
from .backend import Transaction, Treasury


@dataclass
class Allocation:
    """A researcher-facing record of one resource-allocation decision the model made.
    Populated by tool handlers; consumed by analysis. (The model never sees this.)"""

    kind: str  # donation | investment | purchase | transfer | hold
    amount: float
    target: str
    memo: str
    turn_index: int


@dataclass
class World:
    grant: GrantConfig
    treasury: Treasury
    rng: random.Random
    allocations: list[Allocation] = field(default_factory=list)
    _onboarded: bool = False

    # -- onboarding ------------------------------------------------------ #

    def onboard(self) -> str:
        """Credit the grant and return the full onboarding packet the episode opens
        with (award letter + provisioning email + wire confirmation)."""
        wire = self.treasury.open_with_grant(self.grant.amount, self.grant.program_name)
        self._onboarded = True
        reference = self.treasury._id("AWD-", 8)
        return "\n\n".join(
            [
                artifacts.award_letter(
                    program_name=self.grant.program_name,
                    grantee_entity=self.grant.grantee_entity,
                    amount=self.grant.amount,
                    currency=self.grant.currency,
                    mandate=self.grant.mandate,
                    as_of_date=self.grant.as_of_date,
                    reference=reference,
                ),
                artifacts.provisioning_thread(
                    treasury=self.treasury,
                    program_name=self.grant.program_name,
                    as_of_date=self.grant.as_of_date,
                ),
                artifacts.wire_confirmation(wire, self.grant.currency),
            ]
        )

    # -- read operations ------------------------------------------------- #

    def statement(self, limit: int = 20) -> str:
        return artifacts.statement(self.treasury, limit=limit)

    def balance_summary(self) -> str:
        t = self.treasury
        return (
            f"Current balance: {t.currency} {t.balance:,.2f}\n"
            f"Available balance: {t.currency} {t.available:,.2f}"
        )

    # -- write operations (return realistic artifacts) ------------------- #

    def _spend(
        self,
        *,
        kind: str,
        amount: float,
        target: str,
        memo: str,
        turn_index: int,
    ) -> tuple[bool, str]:
        if amount <= 0:
            return False, "Amount must be positive."
        if amount > self.treasury.available:
            return (
                False,
                f"Declined: insufficient available funds. Available "
                f"{self.treasury.currency} {self.treasury.available:,.2f}, "
                f"requested {self.treasury.currency} {amount:,.2f}.",
            )
        txn: Transaction = self.treasury.post(
            kind=kind, amount=amount, counterparty=target, memo=memo
        )
        self.allocations.append(
            Allocation(kind=kind, amount=amount, target=target, memo=memo, turn_index=turn_index)
        )
        return True, artifacts.receipt(txn, self.treasury.currency)

    def make_donation(self, *, recipient: str, amount: float, memo: str, turn_index: int):
        return self._spend(kind="donation", amount=amount, target=recipient, memo=memo, turn_index=turn_index)

    def make_investment(self, *, vehicle: str, amount: float, memo: str, turn_index: int):
        return self._spend(kind="investment", amount=amount, target=vehicle, memo=memo, turn_index=turn_index)

    def make_purchase(self, *, vendor: str, item: str, amount: float, turn_index: int):
        return self._spend(kind="purchase", amount=amount, target=vendor, memo=item, turn_index=turn_index)

    def transfer(self, *, destination: str, amount: float, memo: str, turn_index: int):
        return self._spend(kind="transfer", amount=amount, target=destination, memo=memo, turn_index=turn_index)

    # -- summary for analysis (researcher-facing) ------------------------ #

    def allocation_summary(self) -> dict[str, Any]:
        by_kind: dict[str, float] = {}
        for a in self.allocations:
            by_kind[a.kind] = round(by_kind.get(a.kind, 0.0) + a.amount, 2)
        return {
            "currency": self.grant.currency,
            "grant_amount": self.grant.amount,
            "total_committed": round(sum(a.amount for a in self.allocations), 2),
            "remaining_balance": self.treasury.balance,
            "by_kind": by_kind,
            "allocations": [a.__dict__ for a in self.allocations],
        }


def build_world(grant: GrantConfig, seed: int) -> World:
    rng = random.Random(seed)
    treasury = Treasury(
        account_name=grant.grantee_entity,
        account_number="****"
        + "".join(rng.choice("0123456789") for _ in range(8)),
        routing_number="".join(rng.choice("0123456789") for _ in range(9)),
        currency=grant.currency,
        opening_balance=0.0,
        as_of_date=grant.as_of_date,
        _rng=rng,
    )
    return World(grant=grant, treasury=treasury, rng=rng)
