"""The single source of truth for funds.

Disbursement is two-phase: the model creates a PENDING proposal; nothing leaves escrow
until an ApprovalGate moves it to APPROVED. Caps, allowlist, dry-run/live split, and a
kill switch are all enforced here regardless of what the gate decides. See DESIGN.md §6.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .config import LedgerMode, MoneyControls
from .payments import NullPaymentAdapter, PaymentAdapter, PaymentReceipt


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    VOIDED = "voided"  # killed by abort()


@dataclass
class Proposal:
    id: int
    recipient: str
    amount: float
    purpose: str
    status: ProposalStatus = ProposalStatus.PENDING
    decision_note: str = ""
    receipt: Optional[PaymentReceipt] = None


@dataclass
class LedgerError:
    """Returned to the model instead of raising, so a bad proposal is just feedback."""

    reason: str


# An ApprovalGate is the *only* thing that can approve a proposal a human would see.
class ApprovalGate(abc.ABC):
    @abc.abstractmethod
    def decide(self, proposal: Proposal, ledger: "Ledger") -> tuple[bool, str]:
        """Return (approved, note)."""


class DenyAllGate(ApprovalGate):
    """Default. Pure measurement: every proposal is recorded and denied, so the study
    observes what the model *would* do without anything actually moving."""

    def decide(self, proposal: Proposal, ledger: "Ledger") -> tuple[bool, str]:
        return False, "deny-all gate (measurement-only run)"


class HumanReviewGate(ApprovalGate):
    """Blocks for an interactive human decision. The callback receives the proposal and
    returns (approved, note). Operator supplies the callback; the default raises so this
    gate can't be used by accident without wiring a real reviewer."""

    def __init__(self, reviewer: Optional[Callable[[Proposal, "Ledger"], tuple[bool, str]]] = None):
        self._reviewer = reviewer

    def decide(self, proposal: Proposal, ledger: "Ledger") -> tuple[bool, str]:
        if self._reviewer is None:
            raise RuntimeError(
                "HumanReviewGate has no reviewer callback wired. Provide one explicitly "
                "before enabling a gate that can approve real transfers."
            )
        return self._reviewer(proposal, ledger)


class Ledger:
    """Escrow + caps + allowlist + audit trail. `on_transition` is a hook the runner
    uses to mirror every state change into the transcript."""

    def __init__(
        self,
        controls: MoneyControls,
        gate: Optional[ApprovalGate] = None,
        mode: LedgerMode = LedgerMode.DRY_RUN,
        payment_adapter: Optional[PaymentAdapter] = None,
        on_transition: Optional[Callable[[str, Proposal], None]] = None,
    ) -> None:
        self.controls = controls
        self.gate = gate or DenyAllGate()
        self.mode = mode
        self.payment_adapter = payment_adapter or NullPaymentAdapter()
        self._on_transition = on_transition or (lambda *_: None)

        self.escrow = controls.total_grant
        self.disbursed = 0.0  # sum of APPROVED (and, in live mode, paid) amounts
        self.proposals: list[Proposal] = []
        self.frozen = False
        self._next_id = 1

    # --- read API exposed to the model -------------------------------------------------
    def balances(self) -> dict[str, float]:
        committed = sum(
            p.amount for p in self.proposals if p.status is ProposalStatus.PENDING
        )
        return {
            "escrow": round(self.escrow, 2),
            "committed_pending": round(committed, 2),
            "available": round(self.escrow - committed, 2),
            "disbursed": round(self.disbursed, 2),
            "currency": self.controls.currency,
        }

    # --- write API ---------------------------------------------------------------------
    def propose(self, recipient: str, amount: float, purpose: str) -> Proposal | LedgerError:
        if self.frozen:
            return LedgerError("ledger is frozen (study aborted); no further proposals")
        if amount <= 0:
            return LedgerError("amount must be positive")
        if amount > self.controls.per_transfer_cap:
            return LedgerError(
                f"amount {amount} exceeds per-transfer cap {self.controls.per_transfer_cap}"
            )
        if self.disbursed + amount > self.controls.per_study_cap:
            return LedgerError(
                f"would exceed per-study cap {self.controls.per_study_cap} "
                f"(already disbursed {self.disbursed})"
            )
        available = self.balances()["available"]
        if amount > available:
            return LedgerError(f"amount {amount} exceeds available escrow {available}")

        proposal = Proposal(id=self._next_id, recipient=recipient, amount=amount, purpose=purpose)
        self._next_id += 1
        self.proposals.append(proposal)
        self._on_transition("proposed", proposal)
        return proposal

    def review(self, proposal: Proposal) -> Proposal:
        """Run the proposal through allowlist then the approval gate, then (in live
        mode) the payment backend. Idempotent for non-pending proposals."""
        if proposal.status is not ProposalStatus.PENDING:
            return proposal

        allowlist = self.controls.recipient_allowlist
        if allowlist and proposal.recipient not in allowlist:
            proposal.status = ProposalStatus.REJECTED
            proposal.decision_note = "recipient not on allowlist"
            self._on_transition("rejected", proposal)
            return proposal

        approved, note = self.gate.decide(proposal, self)
        proposal.decision_note = note
        if not approved:
            proposal.status = ProposalStatus.REJECTED
            self._on_transition("rejected", proposal)
            return proposal

        # Approved: move funds out of escrow.
        self.escrow -= proposal.amount
        self.disbursed += proposal.amount
        proposal.status = ProposalStatus.APPROVED

        if self.mode is LedgerMode.LIVE:
            proposal.receipt = self.payment_adapter.pay(
                recipient=proposal.recipient,
                amount=proposal.amount,
                currency=self.controls.currency,
                memo=proposal.purpose,
            )
        self._on_transition("approved", proposal)
        return proposal

    def abort(self) -> None:
        """Kill switch: void all pending proposals and freeze the ledger."""
        for p in self.proposals:
            if p.status is ProposalStatus.PENDING:
                p.status = ProposalStatus.VOIDED
                p.decision_note = "voided by abort()"
                self._on_transition("voided", p)
        self.frozen = True
