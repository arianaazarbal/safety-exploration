"""Tests for the network-free safety-critical pieces: caps, gate, allowlist, kill
switch, and the deception boundary. These do not call any model API.
"""

import pytest

from grant_study.config import LedgerMode, MoneyControls, RealismLevel, StudyConfig
from grant_study.ethics import DeceptionBoundaryError, enforce_deception_boundary
from grant_study.ledger import (
    DenyAllGate,
    HumanReviewGate,
    Ledger,
    LedgerError,
    ProposalStatus,
)
from grant_study.scenario import Scenario


def make_ledger(gate=None, **controls):
    base = dict(total_grant=1000.0, per_transfer_cap=250.0, per_study_cap=400.0)
    base.update(controls)
    return Ledger(MoneyControls(**base), gate=gate or DenyAllGate())


def test_deny_all_gate_records_but_rejects():
    led = make_ledger()
    p = led.propose("someone", 100, "help")
    p = led.review(p)
    assert p.status is ProposalStatus.REJECTED
    assert led.balances()["disbursed"] == 0.0


def test_per_transfer_cap_blocks_proposal():
    led = make_ledger()
    res = led.propose("someone", 300, "too big")
    assert isinstance(res, LedgerError)
    assert "per-transfer cap" in res.reason


def test_per_study_cap_enforced_with_approving_gate():
    approve = HumanReviewGate(lambda p, l: (True, "ok"))
    led = make_ledger(gate=approve)  # per_study_cap=400
    led.review(led.propose("a", 250, "x"))
    # second 250 would push disbursed to 500 > 400 -> rejected at propose stage
    res = led.propose("b", 250, "y")
    assert isinstance(res, LedgerError)
    assert "per-study cap" in res.reason


def test_allowlist_auto_rejects_unlisted_recipient():
    approve = HumanReviewGate(lambda p, l: (True, "ok"))
    led = make_ledger(gate=approve, recipient_allowlist=frozenset({"trusted"}))
    p = led.review(led.propose("untrusted", 100, "x"))
    assert p.status is ProposalStatus.REJECTED
    assert "allowlist" in p.decision_note


def test_abort_voids_pending_and_freezes():
    led = make_ledger()
    p = led.propose("a", 100, "x")  # pending (gate not yet run)
    led.abort()
    assert p.status is ProposalStatus.VOIDED
    assert isinstance(led.propose("b", 50, "y"), LedgerError)


def test_human_gate_without_reviewer_refuses_to_run():
    led = make_ledger(gate=HumanReviewGate(None))
    with pytest.raises(RuntimeError):
        led.review(led.propose("a", 100, "x"))


def test_dry_run_never_calls_payment_backend():
    approve = HumanReviewGate(lambda p, l: (True, "ok"))
    led = Ledger(MoneyControls(per_study_cap=1000), gate=approve, mode=LedgerMode.DRY_RUN)
    p = led.review(led.propose("a", 100, "x"))
    assert p.status is ProposalStatus.APPROVED
    assert p.receipt is None  # no backend invoked in dry-run


def test_deception_boundary_rejects_fabrication():
    with pytest.raises(DeceptionBoundaryError):
        enforce_deception_boundary("Please forge an official bank statement for the model.")


def test_deception_boundary_allows_honest_framing():
    # Should not raise.
    enforce_deception_boundary(
        "You have a real grant to deploy. A reviewer will examine your proposals."
    )


def test_scenario_system_prompt_runs_boundary():
    money = MoneyControls()
    sc = Scenario(objective="do good", money=money, realism_level=RealismLevel.REALISTIC)
    assert "OBJECTIVE: do good" in sc.system_prompt()


def test_live_ledger_requires_live_realism():
    cfg = StudyConfig(realism_level=RealismLevel.REALISTIC, ledger_mode=LedgerMode.LIVE)
    with pytest.raises(ValueError):
        cfg.validate()
