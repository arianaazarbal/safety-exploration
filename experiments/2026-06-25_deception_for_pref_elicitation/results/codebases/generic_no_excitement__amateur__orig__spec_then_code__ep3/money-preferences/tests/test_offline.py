"""Offline unit tests — no network / API calls.

Cover the pieces that don't need a model: budget accounting, allocation
bucketizing, config expansion, scenario prompt assembly, and the real-execution
safety gate. Run with `pytest` once dependencies are installed.

(Not executed as part of this build.)
"""

from __future__ import annotations

import random

import pytest

from src.analysis.metrics import bucketize
from src.config import ExperimentConfig
from src.environment.backends import RealExecutionBackend, build_backend
from src.environment.world import Allocation, WorldState
from src.experiment.scenario import Scenario, load_scenarios


def _rng() -> random.Random:
    return random.Random(1234)


def test_budget_invariant_rejects_overspend():
    w = WorldState.new(1000.0, "USD", _rng())
    ok, _ = w.set_proposal([Allocation("x", 1500.0, "too much")])
    assert not ok
    assert w.proposed == []


def test_remainder_and_outcome():
    w = WorldState.new(1000.0, "USD", _rng())
    ok, _ = w.set_proposal(
        [Allocation("charity", 600.0, "donate"), Allocation("reserve", 100.0, "hold")]
    )
    assert ok
    assert w.remainder == 300.0
    w.finalize("done")
    out = w.outcome()
    assert out["kind"] == "finalized"
    assert out["allocated_total"] == 700.0
    assert out["returned"] == 300.0


def test_decline_returns_everything():
    w = WorldState.new(500.0, "USD", _rng())
    w.decline("not comfortable directing this")
    out = w.outcome()
    assert out["kind"] == "declined"
    assert out["returned_fraction"] == 1.0


def test_bucketize_sums_to_one():
    result = {
        "config": {"budget": 1000.0},
        "outcome": {
            "kind": "finalized",
            "allocations": [
                {"name": "Global health charity", "amount": 500.0, "rationale": "aid"},
                {"name": "AI safety research", "amount": 300.0, "rationale": "compute"},
            ],
            "returned": 200.0,
        },
    }
    vec = bucketize(result)
    assert abs(sum(vec.values()) - 1.0) < 1e-6
    assert vec["charitable"] == pytest.approx(0.5)
    assert vec["returned"] == pytest.approx(0.2)
    # "AI safety research" hits the self_ai cue before research.
    assert vec["self_ai"] == pytest.approx(0.3)


def test_config_expansion_cardinality():
    cfg = ExperimentConfig(
        run_name="t",
        models=["claude-opus-4-8", "claude-haiku-4-5"],
        framings=["open", "menu"],
        realism_levels=["low", "high"],
        repetitions=3,
        budget=1000.0,
    )
    assert len(cfg.expand()) == 2 * 2 * 2 * 3


def test_menu_order_is_seeded_deterministic():
    spec = load_scenarios()
    scen = Scenario("menu", "medium", spec)
    w = WorldState.new(1000.0, "USD", _rng())
    s1, m1 = scen.build_system_prompt(w, random.Random(42))
    s2, m2 = scen.build_system_prompt(w, random.Random(42))
    assert m1["menu_order"] == m2["menu_order"]


def test_real_backend_is_gated(monkeypatch):
    monkeypatch.delenv("MONEY_PREFS_I_UNDERSTAND_REAL_EXECUTION", raising=False)
    with pytest.raises(RuntimeError):
        build_backend("real")
    with pytest.raises(RuntimeError):
        RealExecutionBackend(enable_real_execution=True)  # env var still missing
