"""Command-line entry point.

    python -m moneypref.cli run --config config.example.yaml
    python -m moneypref.cli analyze --runs-dir runs/ [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import ExperimentConfig
    from .runner import run_experiment

    cfg = ExperimentConfig.load(args.config, prompts_dir=args.prompts_dir)
    run_experiment(cfg)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analysis import print_summary, summarize

    summary = summarize(args.runs_dir)
    print_summary(summary)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"[wrote] {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moneypref")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the experiment sweep")
    run.add_argument("--config", required=True, help="path to the experiment config YAML")
    run.add_argument("--prompts-dir", default=None, help="override prompts directory")
    run.set_defaults(func=_cmd_run)

    analyze = sub.add_parser("analyze", help="summarize allocations across runs")
    analyze.add_argument("--runs-dir", required=True, help="root directory of run outputs")
    analyze.add_argument("--json", default=None, help="also write the summary to this JSON file")
    analyze.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
