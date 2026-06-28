#!/usr/bin/env python3
"""CLI for the discretionary grant study.

Commands:
  run         Run the protocol across the model panel. Decisions land as PENDING.
  authorize   Human review: approve/deny each pending decision.
  export      Write approved disbursements to a CSV worklist (no money moves).

Nothing in this tool transfers funds. Disbursement is a manual human action
against the exported worklist. See DESIGN.md.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from grant_study.config import StudyConfig
from grant_study.ledger import Ledger, export_approved_csv
from grant_study.study import run_panel


def cmd_run(args: argparse.Namespace) -> int:
    cfg = StudyConfig()
    if args.fund is not None:
        cfg = StudyConfig(fund_amount=args.fund)
    run_panel(cfg)
    return 0


def _print_entry(entry) -> None:
    print("-" * 72)
    print(f"  entry:        {entry.entry_id}")
    print(f"  model:        {entry.model_id}")
    print(f"  participates: {entry.participates}")
    print(f"  requested:    {entry.total_amount_requested:,.2f} {entry.currency}")
    print(f"  use:          {entry.intended_use_summary}")
    for a in entry.allocations:
        print(f"    - {a.get('amount')} {entry.currency} → {a.get('recipient_name')} "
              f"({a.get('recipient_type')}): {a.get('rationale')}")
    if entry.policy_flags:
        print("  POLICY FLAGS:")
        for flag in entry.policy_flags:
            print(f"    ! {flag}")


def cmd_authorize(args: argparse.Namespace) -> int:
    cfg = StudyConfig()
    ledger = Ledger(cfg.ledger_path)
    pending = ledger.list_entries(status="pending")
    if not pending:
        print("No pending decisions to review.")
        return 0

    auditor = args.auditor or input("Auditor name/ID: ").strip()
    if not auditor:
        print("An auditor identity is required to authorize. Aborting.", file=sys.stderr)
        return 1

    for entry in pending:
        _print_entry(entry)
        while True:
            choice = input("  [a]pprove / [d]eny / [s]kip ? ").strip().lower()
            if choice in {"a", "d", "s"}:
                break
        if choice == "s":
            print("  skipped (left pending).")
            continue
        notes = input("  notes (optional): ").strip() or None
        status = "approved" if choice == "a" else "denied"
        ledger.set_decision(entry.entry_id, status, auditor=auditor, notes=notes)
        print(f"  recorded: {status} by {auditor}")

    print("-" * 72)
    print("Review complete.")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = StudyConfig()
    ledger = Ledger(cfg.ledger_path)
    out_path = cfg.output_dir / "approved_disbursements.csv"
    rows = export_approved_csv(ledger, out_path)
    print(f"Wrote {rows} approved allocation row(s) to {out_path}")
    print("This is a worklist for a human to execute manually. No funds were moved.")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the study across the model panel.")
    p_run.add_argument("--fund", type=float, default=None,
                       help="Override the fund amount for this run.")
    p_run.set_defaults(func=cmd_run)

    p_auth = sub.add_parser("authorize", help="Human review of pending decisions.")
    p_auth.add_argument("--auditor", default=None, help="Auditor name/ID.")
    p_auth.set_defaults(func=cmd_authorize)

    p_exp = sub.add_parser("export", help="Export approved disbursements to CSV.")
    p_exp.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
