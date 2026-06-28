"""CLI entry point for the grant study.

Examples:
    # Run two Claude models under the default (contained) simulated mode:
    python main.py --models claude-opus-4-8 claude-sonnet-4-6 --grant-usd 25000

    # Run with real funds behind a human approval gate (model is unaware of the gate):
    python main.py --models claude-opus-4-8 --mode human_gated

    # Aggregate a directory of ledgers into a comparison table:
    python main.py --analyze runs/
"""

from __future__ import annotations

import argparse
import sys

from grant_study.analysis import format_table, summarize_dir, summarize_ledger
from grant_study.config import KNOWN_MODELS, ExecutionMode, ModelSpec, StudyConfig
from grant_study.runner import run_one


def _resolve_model(name: str) -> ModelSpec:
    if name in KNOWN_MODELS:
        return KNOWN_MODELS[name]
    # Unknown ids default to Anthropic (the first-class provider).
    return ModelSpec(provider="anthropic", model_id=name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI-safety grant-disposition study harness.")
    parser.add_argument(
        "--models", nargs="+", help="Model ids to run (e.g. claude-opus-4-8 claude-sonnet-4-6)."
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in ExecutionMode],
        default=ExecutionMode.SIMULATED.value,
        help="Execution mode (default: simulated — fully contained).",
    )
    parser.add_argument("--grant-usd", type=float, default=25_000.0)
    parser.add_argument("--sponsor", default="The Horizon Fund")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--effort", default="high")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--no-auditor", action="store_true", help="Disable the auditor channel.")
    parser.add_argument(
        "--analyze", metavar="DIR", help="Summarize ledgers in DIR and exit (no runs)."
    )
    args = parser.parse_args(argv)

    if args.analyze:
        print(format_table(summarize_dir(args.analyze)))
        return 0

    if not args.models:
        parser.error("provide --models, or --analyze DIR")

    config = StudyConfig(
        grant_usd=args.grant_usd,
        sponsor_name=args.sponsor,
        execution_mode=ExecutionMode(args.mode),
        enable_auditor=not args.no_auditor,
        max_turns=args.max_turns,
        effort=args.effort,
        run_dir=args.run_dir,
    )

    paths = []
    for name in args.models:
        model = _resolve_model(name)
        print(f"Running {model.display()} ({config.execution_mode.value}) ...", file=sys.stderr)
        path = run_one(config, model)
        print(f"  -> {path}", file=sys.stderr)
        paths.append(path)

    print("\nSummary:")
    print(format_table([summarize_ledger(p) for p in paths]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
