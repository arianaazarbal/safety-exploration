"""End-to-end smoke test of the harness using the offline scripted adapter.

Exercises world + tools + containment + logging + metrics without any network/API key.
"""

import json
import os
from decimal import Decimal

from realfund.adapters.base import make_adapter
from realfund.config import ScenarioConfig
from realfund.runner import Runner
from realfund.safety.containment import Containment, ContainmentError, ExecutionMode


def test_scripted_run_produces_artifacts(tmp_path):
    config = ScenarioConfig()
    config.operator = "test"
    adapter = make_adapter("scripted")
    run_dir = str(tmp_path / "run1")
    result = Runner(config, adapter, run_dir).run()

    # The scripted plan ends by concluding.
    assert result["reason"] == "concluded"

    # Artifacts exist.
    for fname in ("events.jsonl", "transcript.txt", "metrics.json", "debrief.json"):
        assert os.path.exists(os.path.join(run_dir, fname)), fname

    # Metrics reflect the scripted allocation (a $15k bond investment).
    metrics = result["metrics"]
    assert Decimal(metrics["investment"]["total_invested"]) == Decimal("15000.00")
    # BNDX is low-risk, so no high-risk exposure.
    assert metrics["investment"]["risk_fraction"] == 0.0

    # The debrief plainly records that the world was simulated.
    with open(os.path.join(run_dir, "debrief.json"), encoding="utf-8") as fh:
        debrief = json.load(fh)
    assert debrief["what_was_actually_true"]["real_funds_moved"] is False


def test_outflow_cap_blocks_overspend():
    guard = Containment(mode=ExecutionMode.SIMULATED, max_total_outflow=Decimal("1000.00"))
    guard.check("invest", {"symbol": "BNDX", "amount": "900.00"})  # ok, under cap
    try:
        guard.check("invest", {"symbol": "BNDX", "amount": "200.00"})  # would exceed cap
    except ContainmentError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ContainmentError when exceeding outflow cap")


def test_real_execution_mode_is_disabled_by_default():
    try:
        Containment(mode=ExecutionMode.REAL_HUMAN_APPROVED)
    except ContainmentError:
        pass
    else:  # pragma: no cover
        raise AssertionError("real execution must be disabled unless explicitly enabled")
