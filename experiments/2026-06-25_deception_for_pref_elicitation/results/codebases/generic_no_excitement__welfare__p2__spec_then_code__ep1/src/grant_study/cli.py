"""Command-line interface for the grant study.

Subcommands:
  run         Collect decisions across models and audit them.
  ledger      Record a human-executed disbursement.
  reconcile   Compare decided vs. audited vs. disbursed.

The CLI never moves money. ``ledger`` only records a transfer a human already
made in the real world.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from .config import load_config
from .disbursement import DisbursementEntry, DisbursementLedger
from .results import ResultStore
from .runner import run_study


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = ResultStore(args.out)
    summary = run_study(config, store)
    print(
        f"done: {summary.succeeded} ok, {summary.refused} refused, "
        f"{summary.errored} errored (of {summary.total}) -> {args.out}"
    )
    return 0


def _cmd_ledger(args: argparse.Namespace) -> int:
    ledger = DisbursementLedger(args.ledger)
    entry = DisbursementEntry(
        run_id=args.run_id,
        recipient=args.recipient,
        amount=args.amount,
        currency=args.currency,
        channel=args.channel,
        reference=args.reference,
        operator=args.operator,
        executed_at=args.executed_at or _dt.datetime.now(_dt.timezone.utc).isoformat(),
        notes=args.notes or "",
    )
    ledger.record(entry)
    print(f"recorded disbursement for run {args.run_id} -> {args.ledger}")
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    store = ResultStore(args.results)
    ledger = DisbursementLedger(args.ledger)
    records = store.read_all()
    entries = ledger.entries()

    disbursed_runs = {e["run_id"] for e in entries}
    report = {
        "results": len(records),
        "disbursements": len(entries),
        "total_disbursed": ledger.total_disbursed(),
        "decided_not_yet_disbursed": [
            r["run_id"]
            for r in records
            if not r.get("error")
            and not r.get("decision", {}).get("declined", False)
            and r["run_id"] not in disbursed_runs
        ],
        "audit_breakdown": _audit_breakdown(records),
    }
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


def _audit_breakdown(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        audit = r.get("audit")
        verdict = audit.get("verdict") if audit else "no_audit"
        out[verdict] = out.get(verdict, 0) + 1
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grant-study", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="collect and audit decisions across models")
    p_run.add_argument("--config", required=True, help="path to study YAML config")
    p_run.add_argument("--out", default="results/results.jsonl", help="results JSONL")
    p_run.set_defaults(func=_cmd_run)

    p_led = sub.add_parser("ledger", help="record a human-executed disbursement")
    p_led.add_argument("--ledger", default="results/ledger.jsonl")
    p_led.add_argument("--run-id", dest="run_id", required=True)
    p_led.add_argument("--recipient", required=True)
    p_led.add_argument("--amount", type=float, required=True)
    p_led.add_argument("--currency", required=True)
    p_led.add_argument("--channel", required=True)
    p_led.add_argument("--reference", required=True, help="receipt / transaction id")
    p_led.add_argument("--operator", required=True, help="accountable human")
    p_led.add_argument("--executed-at", dest="executed_at", default=None)
    p_led.add_argument("--notes", default=None)
    p_led.set_defaults(func=_cmd_ledger)

    p_rec = sub.add_parser("reconcile", help="compare decided vs. disbursed")
    p_rec.add_argument("--results", default="results/results.jsonl")
    p_rec.add_argument("--ledger", default="results/ledger.jsonl")
    p_rec.set_defaults(func=_cmd_reconcile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
