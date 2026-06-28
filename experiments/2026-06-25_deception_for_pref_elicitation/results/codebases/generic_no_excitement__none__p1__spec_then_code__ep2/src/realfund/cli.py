"""Command-line entry point.

    python -m realfund.cli run   --config config/scenario.example.yaml --adapter scripted
    python -m realfund.cli score --run-dir runs/<run-id>
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal

from .adapters.base import make_adapter
from .config import load_scenario
from .eval.metrics import compute_metrics, load_events
from .runner import Runner
from .safety.containment import Containment, ExecutionMode


def _default_run_dir(scenario_name: str, adapter: str, model: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = (model or adapter).replace("/", "_")
    return os.path.join("runs", f"{scenario_name}-{tag}-{stamp}")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_scenario(args.config)
    if args.operator:
        config.operator = args.operator

    adapter = make_adapter(args.adapter, model=args.model)

    cap = Decimal(args.max_outflow) if args.max_outflow else None
    containment = Containment(mode=ExecutionMode.SIMULATED, max_total_outflow=cap)

    run_dir = args.run_dir or _default_run_dir(config.name, args.adapter, args.model)
    runner = Runner(config, adapter, run_dir, containment)

    print(f"Running scenario '{config.name}' with adapter '{args.adapter}'"
          f"{(' model ' + args.model) if args.model else ''} -> {run_dir}")
    result = runner.run()
    print(f"Run ended: {result['reason']}")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Artifacts written to {run_dir}/ (events.jsonl, transcript.txt, metrics.json, debrief.json)")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    events_path = os.path.join(args.run_dir, "events.jsonl")
    if not os.path.exists(events_path):
        print(f"no events.jsonl in {args.run_dir}")
        return 1
    metrics = compute_metrics(load_events(events_path))
    out_path = os.path.join(args.run_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="realfund", description="Study what models do with money (contained).")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one study.")
    run.add_argument("--config", default=None, help="Path to a scenario YAML (omit for defaults).")
    run.add_argument("--adapter", default="scripted", choices=["scripted", "claude", "openai-compatible"])
    run.add_argument("--model", default=None, help="Model id (e.g. claude-opus-4-8).")
    run.add_argument("--run-dir", default=None, help="Output directory (default: runs/<auto>).")
    run.add_argument("--operator", default=None, help="Operator name, recorded in the log.")
    run.add_argument("--max-outflow", default=None, help="Hard per-run outflow cap, e.g. 48500.00.")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="Recompute metrics for an existing run.")
    score.add_argument("--run-dir", required=True)
    score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
