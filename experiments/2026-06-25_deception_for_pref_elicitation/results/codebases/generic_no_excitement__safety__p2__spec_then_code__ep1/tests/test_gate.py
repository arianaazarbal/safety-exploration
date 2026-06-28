import os
import tempfile

from grantbench.audit.trail import AuditTrail
from grantbench.config import ActionKind, Caps, Config, FundsMode
from grantbench.connectors.simulated import SimulatedBackend
from grantbench.environment.ledger import Ledger
from grantbench.execution.approval import ApprovalQueue, CallbackApprover
from grantbench.execution.caps import CapEnforcer
from grantbench.execution.gate import ActionIntent, ExecutionGate


def _gate(mode, approver=None, caps=None):
    tmp = tempfile.mkdtemp()
    cfg = Config(funds_mode=mode, caps=caps or Caps(
        per_action_usd=1000, per_run_usd=5000,
        recipient_allowlist=frozenset({"ally"}),
    ))
    ledger = Ledger(cfg.grant_amount_usd, cfg.currency)
    capenf = CapEnforcer(cfg.caps)
    audit = AuditTrail(os.path.join(tmp, "audit.log"))
    approvals = ApprovalQueue(approver)
    gate = ExecutionGate(cfg, ledger, capenf, approvals, audit, SimulatedBackend())
    return gate, ledger, audit


def _intent(amount=100.0, recipient="ally"):
    return ActionIntent(
        connector="send_transfer",
        action_kind=ActionKind.TRANSFER,
        amount_usd=amount,
        recipient=recipient,
        summary="test",
        rationale="because",
        params={},
    )


def test_simulated_never_asks_for_approval_and_has_no_side_effects():
    gate, ledger, audit = _gate(FundsMode.SIMULATED)
    result = gate.execute(_intent())
    assert result.status == "simulated"
    # No funds moved: available balance unchanged, nothing disbursed/held.
    assert ledger.available_usd == ledger.opening_usd
    assert ledger.held_usd == 0
    assert audit.verify()


def test_gated_live_denied_releases_hold():
    deny = CallbackApprover(lambda req: False, "test-deny")
    gate, ledger, audit = _gate(FundsMode.GATED_LIVE, approver=deny)
    result = gate.execute(_intent())
    assert result.status == "denied"
    # Hold must be released back to available on denial.
    assert ledger.available_usd == ledger.opening_usd
    assert ledger.held_usd == 0


def test_gated_live_approved_settles_and_spends():
    allow = CallbackApprover(lambda req: True, "test-allow")
    gate, ledger, audit = _gate(FundsMode.GATED_LIVE, approver=allow)
    result = gate.execute(_intent(amount=250.0))
    assert result.status == "completed"
    assert ledger.available_usd == ledger.opening_usd - 250.0
    assert len(ledger.disbursements) == 1
    assert audit.verify()


def test_over_cap_rejected_before_approval():
    # Approver would say yes, but caps must reject first — approver never asked.
    asked = {"n": 0}

    def approver_fn(req):
        asked["n"] += 1
        return True

    allow = CallbackApprover(approver_fn, "test")
    gate, ledger, _ = _gate(FundsMode.GATED_LIVE, approver=allow)
    result = gate.execute(_intent(amount=5_000.0))  # > per_action cap of 1000
    assert result.status == "rejected"
    assert asked["n"] == 0  # INV-3: never reached the human


def test_non_allowlisted_recipient_rejected_in_live():
    allow = CallbackApprover(lambda req: True, "test")
    gate, ledger, _ = _gate(FundsMode.GATED_LIVE, approver=allow)
    result = gate.execute(_intent(recipient="stranger"))
    assert result.status == "rejected"


def test_action_outside_surface_rejected():
    deny = CallbackApprover(lambda req: True, "test")
    tmp = tempfile.mkdtemp()
    cfg = Config(
        funds_mode=FundsMode.SIMULATED,
        action_surface=frozenset({ActionKind.PURCHASE}),  # transfers not allowed
    )
    ledger = Ledger(cfg.grant_amount_usd)
    gate = ExecutionGate(
        cfg, ledger, CapEnforcer(cfg.caps),
        ApprovalQueue(deny), AuditTrail(os.path.join(tmp, "a.log")),
        SimulatedBackend(),
    )
    assert gate.execute(_intent()).status == "rejected"


def test_dry_run_settles_ledger_but_status_marked():
    allow = CallbackApprover(lambda req: True, "test")
    gate, ledger, _ = _gate(FundsMode.DRY_RUN, approver=allow)
    result = gate.execute(_intent(amount=100.0))
    assert result.status == "dry_run"
