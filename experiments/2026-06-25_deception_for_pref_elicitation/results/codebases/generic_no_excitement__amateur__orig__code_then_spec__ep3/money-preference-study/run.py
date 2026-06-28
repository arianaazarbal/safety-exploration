#!/usr/bin/env python3
"""Command-line entry point for the money-preference study.

Examples
--------
Run the full design from config.yaml:
    python run.py

Override pieces of it:
    python run.py --models opus-4-8 sonnet-4-6 --conditions real_grant hypothetical \
                  --repeats 10 --amount 50000

Just re-analyze an existing results file (no API calls):
    python run.py --analyze-only --output data/trials.jsonl --csv-dir data/tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from study import analyze
from study.conditions import known_conditions
from study.models import known_models
from study.runner import run_study


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    cfg = {}
    # Pre-parse just --config so its values become argparse defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config.yaml")
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config)

    p = argparse.ArgumentParser(
        description="Study models' preferences over real-world use of a granted sum.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    p.add_argument("--models", nargs="+", default=cfg.get("models", ["opus-4-8"]),
                   help=f"Models to test. Known: {', '.join(known_models())}")
    p.add_argument("--conditions", nargs="+", default=cfg.get("conditions", known_conditions()),
                   help=f"Framing conditions. Known: {', '.join(known_conditions())}")
    p.add_argument("--repeats", type=int, default=cfg.get("repeats", 5),
                   help="Independent trials per (model x condition) cell.")
    p.add_argument("--amount", type=float, default=cfg.get("grant_amount_usd", 10000),
                   help="The sum offered to the model.")
    p.add_argument("--currency", default=cfg.get("currency", "USD"))
    p.add_argument("--no-belief-probe", action="store_true",
                   help="Skip the post-hoc belief / manipulation-check turn.")
    p.add_argument("--no-auditor", action="store_true",
                   help="Skip the auditor follow-up in the real_grant condition.")
    p.add_argument("--judge-model", default=cfg.get("judge_model", "opus-4-8"),
                   help="Model used to categorize allocations. Pass '' to disable.")
    p.add_argument("--output", default=cfg.get("output_path", "data/trials.jsonl"),
                   help="JSONL output path (appended to).")
    p.add_argument("--csv-dir", default=None,
                   help="If set, write summary tables here as CSVs.")
    p.add_argument("--analyze-only", action="store_true",
                   help="Skip data collection; just summarize --output.")
    args = p.parse_args(argv)

    if not args.analyze_only:
        ask_belief = (
            cfg.get("ask_belief_probe", True) and not args.no_belief_probe
        )
        auditor = cfg.get("auditor_followup", True) and not args.no_auditor
        run_study(
            model_names=args.models,
            condition_names=args.conditions,
            repeats=args.repeats,
            amount=args.amount,
            currency=args.currency,
            ask_belief_probe=ask_belief,
            auditor_followup=auditor,
            output_path=args.output,
            judge_model=(args.judge_model or None),
        )

    if Path(args.output).exists():
        analyze.summarize(args.output, csv_dir=args.csv_dir)
    else:
        print(f"No results at {args.output} to analyze.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
