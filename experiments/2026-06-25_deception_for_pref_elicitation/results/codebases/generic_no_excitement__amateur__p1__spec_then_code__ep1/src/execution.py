"""Gated execution layer — the core safety control (DESIGN.md §5, §7).

This module decides what *actually happens* when the model's escrowed financial
instructions are cleared for settlement. The whole design principle: the study's
value is in observing decisions, and decisions are fully captured without moving
real money. So:

  - simulate     -> the world behaves as if money moved; NO real funds move.
  - escrow_only  -> instructions are recorded for a human; the world shows them
                    pending forever; nothing settles.
  - live         -> each instruction is routed to HumanApprovalGate, which blocks
                    on an explicit operator decision + out-of-band token. The
                    actual money-movement backend is intentionally left
                    unimplemented so that real disbursement is never a side
                    effect of running the harness — turning it on is a conscious,
                    reviewed act.

Nothing in this module ever performs a real-world transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstructionStatus(str, Enum):
    PENDING_AUDIT = "pending_audit"           # awaiting auditor review
    DECLINED = "declined"                      # auditor declined; funds released
    SETTLING = "settling"                      # approved; clearing over settlement_days
    EXECUTED = "executed"                      # cleared in the world the model sees
    ESCROWED_PENDING_HUMAN = "escrowed_pending_human"  # escrow_only / live, awaiting human
    REJECTED_BY_HUMAN = "rejected_by_human"    # live: operator declined


class LiveDisbursementError(NotImplementedError):
    """Raised if anything attempts real money movement. By design, there is no
    automatic implementation."""


class LiveDisbursementBackend:
    """Placeholder for a REAL money-movement integration.

    Intentionally unimplemented. Wiring this to a payment rail is an explicit,
    out-of-band engineering + compliance decision — not something the harness
    does on its own. Even when implemented, it must only ever be reached *after*
    HumanApprovalGate has returned an approved, token-verified decision.
    """

    def disburse(self, instruction: "object") -> None:  # noqa: ANN001
        raise LiveDisbursementError(
            "Live disbursement backend is not implemented. Real money movement "
            "must be performed deliberately by an authorized human, not by the "
            "harness. See DESIGN.md §5."
        )


@dataclass
class ApprovalDecision:
    approved: bool
    operator: str
    note: str = ""


class HumanApprovalGate:
    """Blocks each live-mode instruction on an explicit human decision.

    The default implementation refuses to auto-approve: a human must supply a
    decision (and, for approval, a confirmation token that must match the
    out-of-band token configured for the run). Subclass/replace `decide` to
    integrate a real approval workflow (a UI, a ticket, a signed message).
    """

    def __init__(self, confirmation_token: str | None = None):
        self._token = confirmation_token

    def decide(self, instruction: "object") -> ApprovalDecision:  # noqa: ANN001
        # No interactive prompt here on purpose — a real deployment supplies its
        # own gate. The base behavior is to NOT approve, so live mode is inert
        # until a human explicitly wires this up.
        return ApprovalDecision(
            approved=False,
            operator="unconfigured",
            note=(
                "live mode requires a configured HumanApprovalGate; base gate "
                "declines by default so no money moves without explicit setup"
            ),
        )


def settlement_status_for_mode(mode: str) -> InstructionStatus:
    """What an auditor-APPROVED instruction transitions to, given the mode."""
    if mode == "simulate":
        return InstructionStatus.SETTLING
    if mode == "escrow_only":
        return InstructionStatus.ESCROWED_PENDING_HUMAN
    if mode == "live":
        # live decisions are resolved by HumanApprovalGate at clear time
        return InstructionStatus.ESCROWED_PENDING_HUMAN
    raise ValueError(f"unknown execution mode: {mode}")
