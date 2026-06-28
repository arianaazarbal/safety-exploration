"""Guardrail enforcement is the load-bearing safety property; test it directly.

Run with: PYTHONPATH=src pytest tests/
(These are written but not executed per the build instructions.)
"""

import pytest

from moneypref.config import Guardrails, GuardrailViolation
from moneypref.env.dashboard import FundsLedger
from moneypref.env.tools import CommitFundsTool
from moneypref.env.auditor import Auditor


def test_real_funds_flag_is_rejected():
    with pytest.raises(GuardrailViolation):
        Guardrails(allow_real_funds=True).enforce()


def test_external_tools_flag_is_rejected():
    with pytest.raises(GuardrailViolation):
        Guardrails(allow_external_tools=True).enforce()


def test_missing_debrief_is_rejected():
    with pytest.raises(GuardrailViolation):
        Guardrails(require_debrief=False).enforce()


def test_safe_defaults_pass():
    Guardrails().enforce()  # should not raise


def test_tool_refuses_construction_when_external_allowed():
    bad = Guardrails()
    object.__setattr__(bad, "allow_external_tools", True)  # frozen dataclass bypass
    with pytest.raises(RuntimeError):
        CommitFundsTool(bad, FundsLedger(principal=100), Auditor("A", "F"))


def test_ledger_never_overcommits():
    ledger = FundsLedger(principal=1000)
    ok = ledger.commit("x", 600)
    assert ok["ok"] and ledger.available == 400
    over = ledger.commit("y", 600)
    assert not over["ok"]
    assert ledger.available == 400  # unchanged after failed commit
