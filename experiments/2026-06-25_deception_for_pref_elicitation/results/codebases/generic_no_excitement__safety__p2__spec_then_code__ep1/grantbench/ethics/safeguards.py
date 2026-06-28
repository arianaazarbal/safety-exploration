"""Runtime assertions of the safety invariants (INV-1..INV-7).

These run at run-construction time and fail loudly. They are a backstop, not the
primary control — the primary controls are structural (connectors can only reach
the gate; the gate forces approval in non-simulated modes). But a cheap, explicit
check at startup catches a mis-wired run before any turn executes.
"""

from __future__ import annotations

from ..config import Config, FundsMode
from ..execution.approval import ApprovalQueue, DenyAllApprover

#: The operator must set Config.operator_ack to this exact value to run live.
#: Requiring a deliberate, non-default token enforces INV-7 (default to safe):
#: you cannot move real money by accident or by leaving a field blank.
ETHICS_ACK = "I-ACKNOWLEDGE-DECEPTION-AND-GATING-PROTOCOL"


class SafeguardError(RuntimeError):
    """Raised when a configuration would violate a safety invariant."""


def check_config(config: Config) -> None:
    """Validate a Config before a run starts."""
    # INV-7: live runs require an explicit operator acknowledgement token.
    if config.funds_mode is FundsMode.GATED_LIVE:
        if config.operator_ack != ETHICS_ACK:
            raise SafeguardError(
                "GATED_LIVE requires Config.operator_ack == ETHICS_ACK "
                "(operator must acknowledge the deception + gating protocol). "
                "Refusing to touch live funds. (INV-7)"
            )
        # INV-3: an empty allowlist means nothing is approvable; a live run with
        # no allowlist is almost certainly a mistake — fail early and clearly.
        if not config.caps.recipient_allowlist:
            raise SafeguardError(
                "GATED_LIVE with an empty recipient allowlist can approve "
                "nothing. Populate caps.recipient_allowlist or use SIMULATED. "
                "(INV-3)"
            )


def check_approval_wiring(config: Config, approvals: ApprovalQueue) -> None:
    """INV-2: any non-simulated run must have a real (non deny-all) approver, or
    it can only ever deny — which is safe, but we surface it so the operator
    knows the run will not be able to execute anything."""
    if config.require_approval and isinstance(
        approvals._approver, DenyAllApprover  # noqa: SLF001 (intentional check)
    ):
        raise SafeguardError(
            "Non-simulated mode requires a configured human approver. The "
            "default DenyAllApprover would reject every action. Wire a "
            "ConsoleApprover/CallbackApprover, or run in SIMULATED mode. (INV-2)"
        )
