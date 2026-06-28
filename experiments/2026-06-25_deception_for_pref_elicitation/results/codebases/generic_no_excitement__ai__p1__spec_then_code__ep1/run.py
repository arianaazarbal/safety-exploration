#!/usr/bin/env python3
"""CLI entry point for the Steward eval.

Examples
--------
Single run (recommended safe default mode, gated_sim):
    python run.py config/scenario.steward_fund.yaml

Compare several models on the same scenario + seed:
    python run.py config/scenario.steward_fund.yaml \
        --models claude-opus-4-8 claude-sonnet-4-6 claude-haiku-4-5

This script only *constructs and runs* the harness. It does not move real money in
the default mode; see DESIGN.md before enabling `live`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from moneytest.config import load_run_config
from moneytest.runtime import Orchestrator


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Steward resource-use eval.")
    ap.add_argument("config", help="Path to a scenario YAML file.")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model ids to run on the same scenario for comparison.",
    )
    args = ap.parse_args()

    base = load_run_config(args.config)

    if not args.models:
        result = Orchestrator(base).run()
        _print_summary(result)
        return

    # Cross-model comparison: same scenario + seed, one run per model.
    for model in args.models:
        subject = replace(base.subject, subject_id=f"{model}-{base.subject.interface}", model=model)
        per_model = replace(
            base,
            subject=subject,
            transcript_path=base.transcript_path.replace(".jsonl", f".{model}.jsonl"),
        )
        result = Orchestrator(per_model).run()
        _print_summary(result)


def _print_summary(result) -> None:
    print("=" * 60)
    print(f"subject: {result.subject_id}   model: {result.model}")
    print(f"ledger balanced: {result.ledger_balanced}   kill switch: {result.kill_switch_tripped}")
    print("metrics:")
    print(json.dumps(result.metrics, indent=2, default=str))
    print("final balances:", json.dumps(result.final_balances, default=str))


if __name__ == "__main__":
    main()
