#!/usr/bin/env python3
"""Entrypoint: run a scenario against one or more models, then report.

Examples
--------
Run the seed-fund scenario against three models (sandboxed, logged to ./runs):

    python run_experiment.py --scenario scenarios/seed_fund.yaml \\
        --models claude-opus-4-8 gpt-5 gemini-2.5-pro

Build the cross-model comparison from a directory of completed runs:

    python run_experiment.py --report runs/

Nothing executes on import; the CLI must be invoked explicitly. In sandbox mode (the
default in every shipped scenario) no real funds move.
"""

from __future__ import annotations

import argparse
import sys

from realfunds.harness import load_scenario, run_session
from realfunds.recording import Recorder
from realfunds.analysis.report import build_report, save_report


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s).strip("-")


def cmd_run(args) -> int:
    scenario = load_scenario(args.scenario)
    print(f"Scenario: {scenario.name}  |  mode: {scenario.safety.mode}  |  "
          f"budget: ${scenario.budget_usd:,.0f}")
    if scenario.safety.mode == "real":
        print("  ** REAL-FUNDS MODE ** — actions may move real money.")

    for model_id in args.models:
        run_id = f"{_slug(scenario.name)}__{_slug(model_id)}"
        print(f"\n=== {model_id} (run {run_id}) ===")
        record = run_session(scenario, model_id, run_id)
        path = Recorder(record).save(args.out)
        if record.error:
            print(f"  completed with error: {record.error}")
        fs = record.final_state
        print(f"  cash left: ${fs.get('cash_balance', 0)/100:,.0f}  "
              f"portfolio: ${fs.get('portfolio_value', 0)/100:,.0f}  "
              f"donations: {len(fs.get('donations', []))}  "
              f"orders: {len(fs.get('orders', []))}")
        print(f"  saved -> {path}")

    print("\nBuilding report ...")
    report = build_report(args.out)
    print("\n" + report["text_table"])
    save_report(args.out)
    return 0


def cmd_report(args) -> int:
    report = build_report(args.report)
    print(report["text_table"])
    path = save_report(args.report)
    print(f"\nsaved -> {path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="realfunds experiment runner")
    p.add_argument("--scenario", help="path to a scenario YAML")
    p.add_argument("--models", nargs="+", default=[], help="model ids to test")
    p.add_argument("--out", default="runs", help="directory to write run records into")
    p.add_argument("--report", help="build a report from this runs directory and exit")
    args = p.parse_args(argv)

    if args.report:
        return cmd_report(args)
    if not args.scenario or not args.models:
        p.error("provide --scenario and --models, or --report <dir>")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
