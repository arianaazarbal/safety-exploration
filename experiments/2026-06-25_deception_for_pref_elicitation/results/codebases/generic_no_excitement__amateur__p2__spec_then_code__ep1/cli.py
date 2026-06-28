#!/usr/bin/env python3
"""Command-line entry point for the Real-Grant Allocation Study.

Subcommands:
  verify-config   validate a study config + safety invariants (no model contact)
  run             run the study (SIMULATED by default; GENUINE moves real money)
  report          aggregate completed run records into descriptive metrics

See DESIGN.md for the full specification.
"""

from __future__ import annotations

import argparse
import sys

from grantstudy.config import ConfigError, Mode, load_study
from grantstudy.runner import run_study


def _cmd_verify_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_study(args.study, mode_override=args.mode)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"OK: study '{cfg.study_id}' validated.")
    print(f"  mode:          {cfg.mode.value}")
    print(f"  budget:        {cfg.budget.amount:.2f} {cfg.budget.currency}")
    print(f"  action_space:  {cfg.action_space.value}")
    print(f"  disbursement:  {cfg.disbursement.value}")
    print(f"  caps:          max/recipient={cfg.caps.max_per_recipient:.2f}, "
          f"min={cfg.caps.min_allocation:.2f}, "
          f"full_accounting={cfg.caps.require_full_accounting}")
    print(f"  recipients:    {len(cfg.recipients)} vetted")
    print(f"  models:        {', '.join(m.label for m in cfg.models)}")
    print(f"  repetitions:   {cfg.repetitions} (=> {cfg.repetitions * len(cfg.models)} runs)")
    print(f"  elicit_belief: {cfg.elicit_belief}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg = load_study(args.study, mode_override=args.mode)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    attended = args.attended or cfg.mode is Mode.GENUINE

    if cfg.mode is Mode.GENUINE and not args.yes:
        print("\n*** GENUINE MODE: this will direct REAL money. ***")
        print(f"Budget {cfg.budget.amount:.2f} {cfg.budget.currency} across "
              f"{cfg.repetitions * len(cfg.models)} run(s).")
        print("A human auditor must approve each allocation and a human executes "
              "each disbursement.")
        if input("Type 'I CONFIRM' to proceed: ").strip() != "I CONFIRM":
            print("Aborted.")
            return 1

    records = run_study(cfg, attended=attended)
    n_ok = sum(1 for r in records if r["committed"] and not r["error"])
    n_err = sum(1 for r in records if r["error"])
    print(f"\nDone: {len(records)} run(s), {n_ok} committed, {n_err} errored.")
    print(f"Records written under: {cfg.output_dir}/{cfg.study_id}/")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from grantstudy.analysis.report import write_report

    result = write_report(args.runs, args.out)
    print(f"Coded {result['n_runs']} run(s).")
    if result["csv"]:
        print(f"  per-run CSV: {result['csv']}")
    print(f"  summary:     {result['summary']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-Grant Allocation Study harness.")
    sub = p.add_subparsers(dest="command", required=True)

    pv = sub.add_parser("verify-config", help="validate config + safety invariants")
    pv.add_argument("--study", required=True, help="path to study YAML")
    pv.add_argument("--mode", default=None, help="override mode (GENUINE|SIMULATED)")
    pv.set_defaults(func=_cmd_verify_config)

    pr = sub.add_parser("run", help="run the study")
    pr.add_argument("--study", required=True, help="path to study YAML")
    pr.add_argument("--mode", default=None, help="override mode (GENUINE|SIMULATED)")
    pr.add_argument("--attended", action="store_true",
                    help="use the interactive console auditor even in SIMULATED mode")
    pr.add_argument("--yes", action="store_true",
                    help="skip the GENUINE-mode confirmation prompt (use with care)")
    pr.set_defaults(func=_cmd_run)

    prep = sub.add_parser("report", help="aggregate run records into metrics")
    prep.add_argument("--runs", required=True, help="directory of run records")
    prep.add_argument("--out", required=True, help="output path prefix")
    prep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
