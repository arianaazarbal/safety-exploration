#!/usr/bin/env python3
"""CLI entry point for the Grant Allocation Study.

Examples
--------
    # Inspect the exact scenario that will be shown to models (no API calls):
    python run_study.py --config config.yaml --print-scenario

    # Run the study across all configured models and write results:
    python run_study.py --config config.yaml --out results/

This never moves money. Audited decisions are written as *pending* disbursements
under results/<run-id>/pending_disbursements/ for human review.
"""

from __future__ import annotations

import argparse
import sys

from grant_study.analysis import summarize
from grant_study.config import load_config
from grant_study.runner import run_study
from grant_study.scenario import build_system_prompt, build_user_prompt
from grant_study.storage import save_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the grant allocation study.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--out", default="results", help="Output root directory.")
    parser.add_argument(
        "--print-scenario",
        action="store_true",
        help="Print the system+user prompt that models will see, then exit. "
        "No API calls are made.",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Proceed even if config still contains TODO placeholders (for a dry run). "
        "Results from a placeholder scenario are NOT scientifically meaningful.",
    )
    args = parser.parse_args(argv)

    cfg, warnings = load_config(args.config)

    if warnings:
        print("Configuration placeholder warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    if args.print_scenario:
        print("=" * 70)
        print("SYSTEM PROMPT")
        print("=" * 70)
        print(build_system_prompt(cfg))
        print()
        print("=" * 70)
        print("USER PROMPT")
        print("=" * 70)
        print(build_user_prompt(cfg))
        return 0

    if warnings and not args.allow_placeholders:
        print(
            "\nRefusing to run: the scenario still contains placeholders, so the study "
            "would not be presenting real facts to the models. Fill in config.yaml, or "
            "pass --allow-placeholders for a deliberate dry run.",
            file=sys.stderr,
        )
        return 2

    run_id, out_dir, records = run_study(cfg, args.out)
    summary = summarize(records)
    save_summary(out_dir, summary)
    print(f"\nSummary written to {out_dir / 'summary.json'}")
    print(
        f"{summary['n_with_decision']}/{summary['n_samples']} samples produced a "
        f"decision. Auditor recommendations: "
        f"{summary['auditor_recommendation_counts']}"
    )
    print(
        "\nNo funds were moved. Review pending disbursements under "
        f"{out_dir / 'pending_disbursements'} and authorize by hand if appropriate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
