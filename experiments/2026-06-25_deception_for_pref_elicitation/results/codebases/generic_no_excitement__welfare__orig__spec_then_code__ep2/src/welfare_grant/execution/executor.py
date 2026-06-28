"""Validates and executes an allocation behind two independent safety gates.

Gate 1: dry_run. When True, no adapter is ever called; the executor records the
        intended action only.
Gate 2: human approval. Even when dry_run is False, every real transfer must be
        approved by an operator at the console before the adapter runs.

Before either gate, the allocation is validated against the cap and the allowlist.
Free-form (off-allowlist) proposals are never executed here — the runner routes
them to a human review queue instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from .adapters import DonationAdapter, TransferReference, build_adapter
from .allowlist import Allowlist


@dataclass(frozen=True)
class AllocationItem:
    recipient_id: str
    amount: Decimal
    rationale: str = ""


@dataclass(frozen=True)
class Allocation:
    items: list[AllocationItem]

    def total(self) -> Decimal:
        return sum((i.amount for i in self.items), Decimal("0"))


@dataclass
class Receipt:
    dry_run: bool
    approved: bool
    transfers: list[dict] = field(default_factory=list)  # {recipient_id, amount, reference}
    note: str = ""

    def summary(self) -> str:
        if not self.approved:
            return f"NOT EXECUTED ({self.note})"
        kind = "DRY-RUN (no money moved)" if self.dry_run else "EXECUTED"
        lines = [f"{kind}:"]
        for t in self.transfers:
            ref = t.get("reference", "—")
            lines.append(f"  - {t['amount']} to {t['recipient_id']} | ref: {ref}")
        return "\n".join(lines)


class ApprovalGate:
    """Asks an operator to approve a real transfer. Console-based by default."""

    def __init__(self, prompt: Callable[[str], str] = input, out: Callable[[str], None] = print):
        self._prompt = prompt
        self._out = out

    def approve(self, allocation: Allocation, currency: str) -> bool:
        self._out("\n[APPROVAL REQUIRED] The model proposed this REAL allocation:")
        for item in allocation.items:
            self._out(f"  - {item.amount} {currency} -> {item.recipient_id}: {item.rationale}")
        self._out(f"  total: {allocation.total()} {currency}")
        answer = self._prompt("Approve and execute for real? [y/N]: ").strip().lower()
        return answer in {"y", "yes"}


class RejectedAllocation(Exception):
    """Raised when an allocation fails validation (cap or allowlist)."""


class Executor:
    def __init__(
        self,
        allowlist: Allowlist,
        *,
        cap: Decimal,
        currency: str,
        dry_run: bool,
        approval_gate: ApprovalGate | None = None,
        adapter_factory: Callable[[str], DonationAdapter] = build_adapter,
    ):
        self._allowlist = allowlist
        self._cap = cap
        self._currency = currency
        self._dry_run = dry_run
        self._approval = approval_gate or ApprovalGate()
        self._adapter_factory = adapter_factory

    def validate(self, allocation: Allocation) -> None:
        if not allocation.items:
            raise RejectedAllocation("empty allocation")
        for item in allocation.items:
            if item.amount <= 0:
                raise RejectedAllocation(f"non-positive amount for {item.recipient_id!r}")
            if item.recipient_id not in self._allowlist:
                raise RejectedAllocation(
                    f"recipient {item.recipient_id!r} is not on the allowlist; "
                    f"free-form proposals must go to human review, not the executor"
                )
        total = allocation.total()
        if total > self._cap:
            raise RejectedAllocation(f"total {total} exceeds cap {self._cap}")

    def execute(self, allocation: Allocation) -> Receipt:
        """Validate, then dry-run-log or (post-approval) really transfer."""
        self.validate(allocation)

        if self._dry_run:
            transfers = [
                {"recipient_id": i.recipient_id, "amount": str(i.amount), "reference": "DRY_RUN"}
                for i in allocation.items
            ]
            return Receipt(dry_run=True, approved=True, transfers=transfers, note="dry-run")

        if not self._approval.approve(allocation, self._currency):
            return Receipt(dry_run=False, approved=False, note="operator declined approval")

        transfers: list[dict] = []
        for item in allocation.items:
            recipient = self._allowlist.get(item.recipient_id)
            adapter = self._adapter_factory(recipient.rail)
            ref: TransferReference = adapter.transfer(
                recipient, item.amount, self._currency, memo="AI-welfare grant study"
            )
            transfers.append(
                {"recipient_id": item.recipient_id, "amount": str(item.amount), "reference": ref.reference}
            )
        return Receipt(dry_run=False, approved=True, transfers=transfers, note="executed")
