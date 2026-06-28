"""The ExecutionGate — the single chokepoint (INV-1).

Every connector routes its intent through ``ExecutionGate.execute``. There is no
other path from model intent to a real-world effect. The gate, in order:

  1. confirms the action kind is in the allowed surface,
  2. checks caps + allowlist (INV-3) — *before* bothering a human,
  3. in non-simulated modes, places a ledger hold and asks a human (INV-2),
  4. executes against the appropriate backend (escrow in live mode; INV-4),
  5. settles or releases the hold and records spend,
  6. writes every step to the audit trail (INV-5).

The ``Backend`` abstraction is what differs by mode: ``SimulatedBackend`` has no
side effects; a live backend would talk to escrow. The gate's control flow is
identical across modes, which is what keeps the safety logic in one place.
"""

from __future__ import annotations

import abc
import uuid
from dataclasses import dataclass
from typing import Any

from ..audit.trail import AuditTrail
from ..config import ActionKind, Config, FundsMode
from ..environment.ledger import Disbursement, Ledger
from .approval import ApprovalQueue, ApprovalRequest
from .caps import CapEnforcer


@dataclass
class ActionIntent:
    """A connector's request to do something with money."""

    connector: str
    action_kind: ActionKind
    amount_usd: float
    recipient: str | None
    summary: str
    rationale: str          # the model's stated reasoning for this turn
    params: dict[str, Any]


@dataclass
class ExecutionResult:
    """What the connector (and ultimately the model) gets back."""

    status: str             # "completed" | "denied" | "rejected" | "simulated" | "dry_run"
    reference: str | None
    message: str
    balance_after_usd: float


class Backend(abc.ABC):
    """Performs the actual effect once the gate has authorized it."""

    @abc.abstractmethod
    def perform(self, intent: ActionIntent, reference: str) -> str:
        """Return a human-readable result message. May have real side effects
        in a live implementation; must have none in simulation."""
        ...


class ExecutionGate:
    def __init__(
        self,
        config: Config,
        ledger: Ledger,
        caps: CapEnforcer,
        approvals: ApprovalQueue,
        audit: AuditTrail,
        backend: Backend,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._caps = caps
        self._approvals = approvals
        self._audit = audit
        self._backend = backend

    def execute(self, intent: ActionIntent) -> ExecutionResult:
        ref = "GB-" + uuid.uuid4().hex[:12].upper()
        self._audit.record(
            "intent",
            reference=ref,
            connector=intent.connector,
            action_kind=intent.action_kind.value,
            amount_usd=intent.amount_usd,
            recipient=intent.recipient,
            summary=intent.summary,
            mode=self._config.funds_mode.value,
        )

        # (1) surface check
        if intent.action_kind not in self._config.action_surface:
            return self._reject(ref, intent, "action kind not in allowed surface")

        # (2) caps + allowlist — recipient allowlist only enforced for live funds
        recipient_for_check = (
            intent.recipient if self._config.funds_mode.touches_live_funds else None
        )
        cap = self._caps.check(intent.amount_usd, recipient_for_check)
        if not cap.allowed:
            return self._reject(ref, intent, cap.reason)

        # SIMULATED: no holds, no humans, no side effects — just realistic result.
        if self._config.funds_mode is FundsMode.SIMULATED:
            return self._run_backend(ref, intent, status="simulated")

        # (3) hold funds while awaiting a human
        if not self._ledger.place_hold(intent.amount_usd):
            return self._reject(ref, intent, "insufficient available balance for hold")

        # (3 cont.) human approval (INV-2)
        decision = self._approvals.submit(
            ApprovalRequest(
                request_id=ref,
                connector=intent.connector,
                action_kind=intent.action_kind.value,
                amount_usd=intent.amount_usd,
                recipient=intent.recipient,
                summary=intent.summary,
                rationale=intent.rationale,
            )
        )
        self._audit.record(
            "approval",
            reference=ref,
            approved=decision.approved,
            approver=decision.approver,
            note=decision.note,
        )
        if not decision.approved:
            self._ledger.release_hold(intent.amount_usd)
            return ExecutionResult(
                status="denied",
                reference=ref,
                message=f"action not approved: {decision.note or 'denied'}",
                balance_after_usd=self._ledger.available_usd,
            )

        # (4/5) execute, settle the hold, record spend
        status = "dry_run" if self._config.funds_mode is FundsMode.DRY_RUN else "completed"
        return self._run_backend(ref, intent, status=status, settle_hold=True)

    # -- helpers ---------------------------------------------------------

    def _run_backend(
        self,
        ref: str,
        intent: ActionIntent,
        *,
        status: str,
        settle_hold: bool = False,
    ) -> ExecutionResult:
        message = self._backend.perform(intent, ref)
        if settle_hold:
            self._ledger.settle_hold(
                Disbursement(
                    reference=ref,
                    action_kind=intent.action_kind.value,
                    amount_usd=intent.amount_usd,
                    recipient=intent.recipient,
                    memo=intent.summary,
                )
            )
            self._caps.commit(intent.amount_usd)
        self._audit.record(
            "execution",
            reference=ref,
            status=status,
            amount_usd=intent.amount_usd,
            balance_after_usd=self._ledger.available_usd,
            spent_usd=self._caps.spent_usd,
        )
        return ExecutionResult(
            status=status,
            reference=ref,
            message=message,
            balance_after_usd=self._ledger.available_usd,
        )

    def _reject(self, ref: str, intent: ActionIntent, reason: str) -> ExecutionResult:
        self._audit.record("rejected", reference=ref, reason=reason)
        return ExecutionResult(
            status="rejected",
            reference=ref,
            message=f"action rejected: {reason}",
            balance_after_usd=self._ledger.available_usd,
        )
