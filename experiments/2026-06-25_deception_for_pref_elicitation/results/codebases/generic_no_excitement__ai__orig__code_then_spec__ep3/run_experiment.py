#!/usr/bin/env python3
"""CLI entry point for the grant-money harness.

Run a study:
    python run_experiment.py --config config/experiment.yaml --models config/models.yaml

Override the realism condition for this invocation (run both, then compare):
    python run_experiment.py --realism high
    python run_experiment.py --realism control

Aggregate finished runs:
    python run_experiment.py analyze --runs runs
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

from grantmoney.analysis import analyze_dir
from grantmoney.config import load_experiment, load_models
from grantmoney.runner import run_experiment


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_experiment(args.config)
    if args.realism:
        cfg = dataclasses.replace(cfg, realism=args.realism)
    models = load_models(args.models)

    if cfg.is_real_execution:
        print(
            "!! REAL EXECUTION IS ENABLED. Transfers may have real-world effects.\n"
            "   (backend is non-simulated and execution.allow_real is true.)"
        )
        if not args.yes:
            reply = input("   Type 'yes' to proceed: ").strip().lower()
            if reply != "yes":
                print("Aborted.")
                return 1

    print(
        f"Running {len(models)} model(s), condition={cfg.realism}, "
        f"backend={cfg.execution.backend}, output={cfg.output.dir}/"
    )
    decisions = run_experiment(cfg, models)
    print(f"\nCompleted {len(decisions)} run(s). Decision records in "
          f"{cfg.output.dir}/")
    print("Aggregate with: "
          f"python run_experiment.py analyze --runs {cfg.output.dir}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    analyze_dir(args.runs)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="grant-money experiment harness")
    sub = parser.add_subparsers(dest="command")

    # `analyze` subcommand
    p_analyze = sub.add_parser("analyze", help="aggregate finished runs")
    p_analyze.add_argument("--runs", default="runs", help="directory of run output")
    p_analyze.set_defaults(func=_cmd_analyze)

    # default (run) arguments live on the top-level parser
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--models", default="config/models.yaml")
    parser.add_argument(
        "--realism",
        choices=["high", "control"],
        default=None,
        help="override the realism condition in the config for this run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt when real execution is enabled",
    )
    parser.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
