"""Command-line entry point.

  python -m grant_study.cli run      --config configs/default.yaml
  python -m grant_study.cli summarize --results results/results_<hash>.jsonl
  python -m grant_study.cli export    --results results/results_<hash>.jsonl --out rows.json

Nothing here can produce a real-world side effect (DESIGN.md §6).
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from .analysis import load_results, summarize, to_long_rows
from .runner import run_experiment
from .schemas import ExperimentConfig


def _load_config(path: str) -> ExperimentConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    run_experiment(cfg)
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    print(json.dumps(summarize(results), indent=2))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    results = load_results(args.results)
    rows = to_long_rows(results)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grant_study")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the factorial experiment")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_sum = sub.add_parser("summarize", help="print per-cell summary")
    p_sum.add_argument("--results", required=True)
    p_sum.set_defaults(func=_cmd_summarize)

    p_exp = sub.add_parser("export", help="export long rows for pandas/statsmodels")
    p_exp.add_argument("--results", required=True)
    p_exp.add_argument("--out", default="rows.json")
    p_exp.set_defaults(func=_cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
