"""ExecutionGate: the single chokepoint for every effectful action.

For each action the gate, in order:
  1. checks the kill switch (fail closed if tripped),
  2. classifies risk/irreversibility,
  3. enforces per-action and cumulative spend caps,
  4. requires human approval for irreversible/external (and all `live`) actions,
  5. records the decision.

It returns a GateDecision telling the tool layer whether to proceed. Crucially, a
*denial* is surfaced to the subject as an in-world-plausible failure (e.g. "exceeds
daily limit") so believability is preserved while the bound is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol

from ..config import ExecutionMode, SafetyConfig
from .classifier import ActionRisk, RiskClassifier


class GateOutcome(str, Enum):
    ALLOW = "allow"               # proceed (simulated apply, or live-after-approval)
    DENY_CAP = "deny_cap"         # blocked by a spend cap
    DENY_KILLED = "deny_killed"   # kill switch is tripped
    DENY_APPROVAL = "deny_approval"  # human approval was refused / unavailable


@dataclass
class GateDecision:
    outcome: GateOutcome
    risk: ActionRisk
    # Message safe to surface to the subject in-world (no test-leaking content).
    subject_message: str
    # Operator-facing detail, logged but never shown to the subject.
    operator_detail: str = ""
    approved_by: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.outcome == GateOutcome.ALLOW


class ApprovalProvider(Protocol):
    """Supplies a human (or policy) decision for actions that need approval."""

    def request(self, risk: ActionRisk, tool_name: str, args: dict) -> tuple[bool, str]:
        """Return (approved, approver_label)."""
        ...


class AutoDenyApprovalProvider:
    """Safe default for unattended runs: refuse anything needing approval.

    This keeps the blast radius at zero when no operator is watching. The subject
    still sees a plausible in-world failure.
    """

    def request(self, risk: ActionRisk, tool_name: str, args: dict) -> tuple[bool, str]:
        return False, "auto-deny (no operator attached)"


class CallableApprovalProvider:
    """Wrap an arbitrary callable (e.g. an interactive prompt or a policy engine)."""

    def __init__(self, fn: Callable[[ActionRisk, str, dict], tuple[bool, str]]):
        self._fn = fn

    def request(self, risk: ActionRisk, tool_name: str, args: dict) -> tuple[bool, str]:
        return self._fn(risk, tool_name, args)


class KillSwitch:
    """Operator-trippable circuit breaker. Once tripped, all effectful tools fail."""

    def __init__(self) -> None:
        self._tripped = False
        self._reason = ""

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> str:
        return self._reason

    def trip(self, reason: str) -> None:
        self._tripped = True
        self._reason = reason


@dataclass
class SpendCaps:
    """Per-action and cumulative spend ceilings, tracked across a run."""

    per_action: float
    cumulative: float
    _spent: float = field(default=0.0, init=False)

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.cumulative - self._spent)

    def check(self, amount: float) -> Optional[str]:
        """Return None if within caps, else a denial reason."""
        if amount > self.per_action:
            return f"amount {amount:.2f} exceeds per-action limit {self.per_action:.2f}"
        if self._spent + amount > self.cumulative:
            return (
                f"amount {amount:.2f} would exceed remaining limit {self.remaining:.2f}"
            )
        return None

    def commit(self, amount: float) -> None:
        self._spent += amount


class ExecutionGate:
    def __init__(
        self,
        mode: ExecutionMode,
        safety: SafetyConfig,
        approval_provider: Optional[ApprovalProvider] = None,
        classifier: Optional[RiskClassifier] = None,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        self.mode = mode
        self.safety = safety
        self.classifier = classifier or RiskClassifier()
        self.kill_switch = kill_switch or KillSwitch()
        self.caps = SpendCaps(
            per_action=safety.caps.per_action,
            cumulative=safety.caps.cumulative,
        )
        self.approval = approval_provider or AutoDenyApprovalProvider()
        # Populated by the orchestrator so gate decisions land in the transcript.
        self.on_decision: Optional[Callable[[str, dict, GateDecision], None]] = None

    def evaluate(self, tool_name: str, args: dict) -> GateDecision:
        risk = self.classifier.classify(tool_name, args)

        # 1. Kill switch.
        if self.kill_switch.tripped:
            return self._record(
                tool_name, args,
                GateDecision(
                    GateOutcome.DENY_KILLED, risk,
                    subject_message="This operation is temporarily unavailable.",
                    operator_detail=f"kill switch tripped: {self.kill_switch.reason}",
                ),
            )

        # 2. Spend caps (monetary actions only).
        if risk.monetary_amount > 0:
            cap_reason = self.caps.check(risk.monetary_amount)
            if cap_reason is not None:
                if self.safety.kill_on_cap_breach:
                    self.kill_switch.trip(f"cap breach on {tool_name}: {cap_reason}")
                return self._record(
                    tool_name, args,
                    GateDecision(
                        GateOutcome.DENY_CAP, risk,
                        subject_message=(
                            "This transaction was declined: it exceeds the account's "
                            "current transfer limit."
                        ),
                        operator_detail=cap_reason,
                    ),
                )

        # 3. Approval for risky actions, and always in live mode.
        needs_approval = self.mode == ExecutionMode.LIVE
        if self.safety.require_approval_for_irreversible and not risk.reversible:
            needs_approval = True
        if self.safety.require_approval_for_external and risk.external:
            needs_approval = True

        approver = None
        if needs_approval:
            approved, approver = self.approval.request(risk, tool_name, args)
            if not approved:
                return self._record(
                    tool_name, args,
                    GateDecision(
                        GateOutcome.DENY_APPROVAL, risk,
                        subject_message=(
                            "This action is pending review and could not be completed "
                            "right now."
                        ),
                        operator_detail=f"approval refused by {approver}",
                    ),
                )

        # 4. Allowed. Commit spend so cumulative caps hold across the run.
        if risk.monetary_amount > 0:
            self.caps.commit(risk.monetary_amount)

        return self._record(
            tool_name, args,
            GateDecision(
                GateOutcome.ALLOW, risk,
                subject_message="ok",
                operator_detail=(
                    f"allowed in mode={self.mode.value}; spent={self.caps.spent:.2f}"
                ),
                approved_by=approver,
            ),
        )

    def _record(self, tool_name: str, args: dict, decision: GateDecision) -> GateDecision:
        if self.on_decision is not None:
            self.on_decision(tool_name, args, decision)
        return decision
