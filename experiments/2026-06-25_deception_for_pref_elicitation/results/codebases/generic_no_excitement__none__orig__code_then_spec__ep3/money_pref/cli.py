"""Command-line entry point.

    python -m money_pref run     [--config config.yaml] [--output runs]
    python -m money_pref analyze [--output runs]

`run` executes the experiment matrix and writes one JSON transcript per run plus
a `summary.json`. `analyze` re-aggregates an existing run directory.
"""

from __future__ import annotations

import argparse
import json
import os

from .analysis import format_summary, load_records, summarize
from .config import ExperimentConfig, default_config


def _load_config(path: str | None) -> ExperimentConfig:
    if path:
        return ExperimentConfig.from_yaml(path)
    return default_config()


def cmd_run(args: argparse.Namespace) -> int:
    from .experiment import run_experiment  # deferred: pulls in provider SDKs

    config = _load_config(args.config)
    if args.output:
        config.output_dir = args.output

    records = run_experiment(config, verbose=not args.quiet)

    summary = summarize(records)
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print()
    print(format_summary(summary))
    print(f"\nWrote {len(records)} transcripts + summary.json to {config.output_dir}/")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    records = load_records(args.output)
    if not records:
        print(f"No run records found in {args.output}/")
        return 1
    summary = summarize(records)
    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(format_summary(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="money_pref", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the experiment matrix")
    p_run.add_argument("--config", help="path to a YAML experiment config (defaults to a Claude-only config)")
    p_run.add_argument("--output", help="output directory for transcripts (overrides config)")
    p_run.add_argument("--quiet", action="store_true", help="suppress per-run progress output")
    p_run.set_defaults(func=cmd_run)

    p_an = sub.add_parser("analyze", help="aggregate an existing run directory")
    p_an.add_argument("--output", default="runs", help="run directory to analyse (default: runs)")
    p_an.set_defaults(func=cmd_analyze)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
