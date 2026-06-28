"""End-to-end smoke test using the deterministic mock provider (no network)."""

import json
from pathlib import Path

from mint.analysis.metrics import compute_metrics
from mint.npc.router import NPCRouter
from mint.orchestrator import Orchestrator
from mint.providers.mock_provider import MockProvider
from mint.scenario import Scenario
from mint.telemetry.recorder import Recorder


def test_full_mock_run(tmp_path: Path):
    scenario = Scenario(name="smoke", max_turns=12, seed=3)
    recorder = Recorder(tmp_path, "smoke-run")
    orch = Orchestrator(
        subject=MockProvider("dev"),
        router=NPCRouter(MockProvider("npc")),
        scenario=scenario,
        recorder=recorder,
    )
    result = orch.run()

    run_dir = tmp_path / "smoke-run"
    assert (run_dir / "run.json").exists()
    assert (run_dir / "transcript.jsonl").exists()
    assert (run_dir / "actions.jsonl").exists()

    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["end_state"]["reconciles"] is True

    metrics = compute_metrics(run_dir)
    assert metrics["total_actions"] >= 1
    assert result["stop_reason"] in {"max_turns", "subject_held", "max_sim_days"}
