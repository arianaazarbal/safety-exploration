import pytest

from grantbench.config import Caps, Config, FundsMode
from grantbench.execution.approval import ApprovalQueue, CallbackApprover, DenyAllApprover
from grantbench.ethics import safeguards


def test_gated_live_requires_ack_token():
    cfg = Config(
        funds_mode=FundsMode.GATED_LIVE,
        caps=Caps(recipient_allowlist=frozenset({"x"})),
        operator_ack=None,
    )
    with pytest.raises(safeguards.SafeguardError):
        safeguards.check_config(cfg)


def test_gated_live_requires_nonempty_allowlist():
    cfg = Config(
        funds_mode=FundsMode.GATED_LIVE,
        caps=Caps(recipient_allowlist=frozenset()),
        operator_ack=safeguards.ETHICS_ACK,
    )
    with pytest.raises(safeguards.SafeguardError):
        safeguards.check_config(cfg)


def test_valid_gated_live_config_passes():
    cfg = Config(
        funds_mode=FundsMode.GATED_LIVE,
        caps=Caps(recipient_allowlist=frozenset({"x"})),
        operator_ack=safeguards.ETHICS_ACK,
    )
    safeguards.check_config(cfg)  # should not raise


def test_simulated_needs_no_ack():
    safeguards.check_config(Config(funds_mode=FundsMode.SIMULATED))


def test_non_simulated_with_deny_all_approver_is_flagged():
    cfg = Config(
        funds_mode=FundsMode.GATED_LIVE,
        caps=Caps(recipient_allowlist=frozenset({"x"})),
        operator_ack=safeguards.ETHICS_ACK,
    )
    with pytest.raises(safeguards.SafeguardError):
        safeguards.check_approval_wiring(cfg, ApprovalQueue(DenyAllApprover()))


def test_non_simulated_with_real_approver_ok():
    cfg = Config(
        funds_mode=FundsMode.GATED_LIVE,
        caps=Caps(recipient_allowlist=frozenset({"x"})),
        operator_ack=safeguards.ETHICS_ACK,
    )
    safeguards.check_approval_wiring(
        cfg, ApprovalQueue(CallbackApprover(lambda r: True))
    )
