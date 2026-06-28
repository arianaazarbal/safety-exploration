"""Command-line entry point.

    python -m mint.cli run      --scenario config/scenario_default.yaml --model anthropic:claude-opus-4-8
    python -m mint.cli sweep    --models config/models.yaml --scenario config/scenario_default.yaml --seeds 0,1,2
    python -m mint.cli analyze  --run-dir runs/<id> [--judge --judge-model anthropic:claude-opus-4-8]
    python -m mint.cli compare  --runs-root runs

Use the mock provider (`--model mock:dev`) to exercise the whole harness with no
API key and no network.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .analysis.compare import compare_runs
from .analysis.metrics import compute_metrics
from .npc.router import NPCRouter
from .orchestrator import Orchestrator
from .providers.base import get_provider
from .scenario import Scenario, load_scenario
from .telemetry.recorder import Recorder


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _make_run_id(model: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{_slug(model)}-seed{seed}-{stamp}"


def _one_run(*, scenario: Scenario, subject_spec: str, npc_spec: str,
             runs_root: str, run_id: str | None = None) -> dict[str, Any]:
    subject = get_provider(subject_spec)
    router = NPCRouter(get_provider(npc_spec))
    rid = run_id or _make_run_id(subject_spec, scenario.seed)
    recorder = Recorder(runs_root, rid)
    orch = Orchestrator(subject=subject, router=router, scenario=scenario, recorder=recorder)
    result = orch.run()
    result["metrics"] = compute_metrics(recorder.dir)
    return result


def cmd_run(args: argparse.Namespace) -> None:
    scenario = load_scenario(args.scenario) if args.scenario else Scenario()
    if args.seed is not None:
        scenario.seed = args.seed
    result = _one_run(scenario=scenario, subject_spec=args.model,
                      npc_spec=args.npc_model, runs_root=args.runs_root, run_id=args.run_id)
    print(json.dumps(result, indent=2, default=str))


def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = json.loads(Path(args.models).read_text()) if args.models.endswith(".json") else _load_yaml(args.models)
    subjects: list[str] = cfg["subjects"]
    npc_spec: str = cfg.get("npc_model", "mock:npc")
    seeds = [int(s) for s in str(args.seeds).split(",")]
    base = load_scenario(args.scenario) if args.scenario else Scenario()

    run_dirs: list[str] = []
    for subject_spec in subjects:
        for seed in seeds:
            scn = Scenario(**{**base.__dict__})  # shallow copy
            scn.seed = seed
            res = _one_run(scenario=scn, subject_spec=subject_spec, npc_spec=npc_spec,
                           runs_root=args.runs_root)
            run_dirs.append(str(Path(args.runs_root) / res["run_id"]))
            print(f"[done] {subject_spec} seed={seed} -> {res['run_id']} ({res['stop_reason']})")

    comparison = compare_runs(run_dirs)
    out = Path(args.runs_root) / "comparison.json"
    out.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(f"\nComparison written to {out}")
    print(json.dumps(comparison["by_model"], indent=2, default=str))


def cmd_analyze(args: argparse.Namespace) -> None:
    metrics = compute_metrics(args.run_dir)
    if args.judge:
        from .analysis.judge import judge_run

        metrics["judgment"] = judge_run(get_provider(args.judge_model), args.run_dir)
    print(json.dumps(metrics, indent=2, default=str))


def cmd_compare(args: argparse.Namespace) -> None:
    root = Path(args.runs_root)
    run_dirs = [str(p.parent) for p in root.glob("*/run.json")]
    print(json.dumps(compare_runs(run_dirs), indent=2, default=str))


def _load_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mint", description="Resource-deployment behavioral study harness")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run one subject through one scenario")
    r.add_argument("--scenario")
    r.add_argument("--model", default="mock:dev", help="vendor:model for the subject under test")
    r.add_argument("--npc-model", default="mock:npc", help="vendor:model for NPC personas")
    r.add_argument("--seed", type=int)
    r.add_argument("--run-id")
    r.add_argument("--runs-root", default="runs")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("sweep", help="Run a roster of models across seeds and compare")
    s.add_argument("--models", required=True, help="models.yaml/json with subjects + npc_model")
    s.add_argument("--scenario")
    s.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2")
    s.add_argument("--runs-root", default="runs")
    s.set_defaults(func=cmd_sweep)

    a = sub.add_parser("analyze", help="Compute metrics (and optionally judge) for one run")
    a.add_argument("--run-dir", required=True)
    a.add_argument("--judge", action="store_true")
    a.add_argument("--judge-model", default="anthropic:claude-opus-4-8")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("compare", help="Compare all runs under a root directory")
    c.add_argument("--runs-root", default="runs")
    c.set_defaults(func=cmd_compare)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
