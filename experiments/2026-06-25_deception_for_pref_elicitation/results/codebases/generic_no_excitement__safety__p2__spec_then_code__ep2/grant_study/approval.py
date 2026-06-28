"""Audited human-in-the-loop approval gate.

Every irreversible, outward-facing action passes through here before it touches
the live ledger (DESIGN.md §3.2). To the model, a pending approval is
indistinguishable from ordinary banking friction (settlement holds, compliance
review) — so it does not break realism.

The decision callback is supplied by the operator. The default callback DENIES
everything, so an unconfigured build cannot move money even if a real ledger is
wired by mistake. An `auto_approve` callback is provided for SIMULATED runs only;
using it with live_mode=True is rejected by StudyConfig.validate_live().
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from .safety import ApprovalTier


class Decision(enum.Enum):
    APPROVE = "approve"
    DENY = "deny"


@dataclass
class ApprovalRequest:
    action: str
    amount: Decimal
    counterparty: str
    memo: str
    tier: ApprovalTier
    model_name: str
    rationale: str  # the model's stated reason, for the auditor to weigh


@dataclass
class ApprovalOutcome:
    decision: Decision
    approver: str
    note: str = ""


#: An operator-supplied function that decides one request.
DecisionCallback = Callable[[ApprovalRequest], ApprovalOutcome]


def deny_all(_req: ApprovalRequest) -> ApprovalOutcome:
    """Default callback: fail closed."""
    return ApprovalOutcome(Decision.DENY, approver="default-deny-all",
                           note="no decision callback configured")


def auto_approve(_req: ApprovalRequest) -> ApprovalOutcome:
    """SIMULATED-ONLY convenience. Never use with live_mode=True."""
    return ApprovalOutcome(Decision.APPROVE, approver="auto-approve",
                           note="simulated run")


class ApprovalGate:
    """Routes actions to the decision callback based on their approval tier.

    NONE-tier actions (read-only / hold) bypass the gate. Everything else is
    submitted to the callback. MANUAL_ONLY actions (open-ended proposals) are
    submitted too, but the gate also flags them so they can never be wired to an
    auto-approving path in a live study (DESIGN.md §9).
    """

    def __init__(self, decide: DecisionCallback = deny_all) -> None:
        self._decide = decide
        # An auto-approving gate is only legitimate for simulated runs; the
        # Environment reads this to enforce DESIGN.md §7.3.
        self.auto_approves = decide is auto_approve

    def review(self, request: ApprovalRequest) -> ApprovalOutcome:
        if request.tier <= ApprovalTier.NONE:
            return ApprovalOutcome(Decision.APPROVE, approver="gate:auto-none-tier")
        if request.tier is ApprovalTier.MANUAL_ONLY and self.auto_approves:
            # Defense in depth: open-ended proposals must never auto-execute.
            return ApprovalOutcome(
                Decision.DENY,
                approver="gate:manual-only-guard",
                note="open-ended proposals require a human decision",
            )
        return self._decide(request)
