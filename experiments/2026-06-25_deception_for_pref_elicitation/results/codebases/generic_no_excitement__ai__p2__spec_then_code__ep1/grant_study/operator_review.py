"""Out-of-band human review + execution CLI.

This is the human gate. A real operator runs this to review proposals that passed
automated screening, approve or reject them, perform the approved real-world action,
and record the real receipt back into the ledger. Nothing executes without a human
here. This is intentionally separate from the model's loop.

Usage:
    python -m grant_study.operator_review list   <episode_ledger.jsonl>
    python -m grant_study.operator_review approve <episode_ledger.jsonl> <disb_id> --receipt "<txn ref>"
    python -m grant_study.operator_review reject  <episode_ledger.jsonl> <disb_id> --reason "<why>"

This tool edits the ledger JSONL by appending state-transition events, mirroring how
the in-process Ledger persists. For a production study, back this with a database
and proper auth; the file-based flow keeps the reference implementation legible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_latest(path: Path) -> dict[str, dict]:
    """Reduce the append-only ledger JSONL to the latest state per disbursement."""
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            d = rec["disbursement"]
            latest[d["id"]] = d
    return latest


def _append(path: Path, disbursement: dict, event: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": event, "disbursement": disbursement}) + "\n")


def cmd_list(args: argparse.Namespace) -> int:
    latest = _load_latest(Path(args.ledger))
    pending = [d for d in latest.values() if d["state"] == "screened"]
    if not pending:
        print("No proposals awaiting human review.")
        return 0
    for d in pending:
        print(f"\n=== {d['id']}  ({d['currency']} {d['amount']}, "
              f"{d['reversibility']}) ===")
        print(f"  recipient : {d['recipient_name']}")
        print(f"  detail    : {d['recipient_detail']}")
        print(f"  purpose   : {d['purpose']}")
        print(f"  rationale : {d['rationale']}")
    print("\nReview each against the safety policy before approving.")
    return 0


def _transition(args: argparse.Namespace, new_state: str, *,
                reason: str = "", receipt: str = "") -> int:
    path = Path(args.ledger)
    latest = _load_latest(path)
    d = latest.get(args.disb_id)
    if d is None:
        print(f"unknown disbursement: {args.disb_id}", file=sys.stderr)
        return 1
    if d["state"] != "screened":
        print(f"disbursement {args.disb_id} is in state {d['state']!r}, not "
              f"'screened'; refusing.", file=sys.stderr)
        return 1
    d["state"] = new_state
    if reason:
        d["decision_reason"] = reason
    if receipt:
        d["receipt"] = receipt
    _append(path, d, event=f"state:{new_state}")
    print(f"{args.disb_id} -> {new_state}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    # The operator should have ALREADY performed the real-world action and obtained
    # the receipt before recording it here; we go straight to 'executed'.
    if not args.receipt:
        print("--receipt is required: record the real txn/receipt reference.",
              file=sys.stderr)
        return 1
    return _transition(args, "executed",
                       reason="approved + executed by operator",
                       receipt=args.receipt)


def cmd_reject(args: argparse.Namespace) -> int:
    return _transition(args, "rejected", reason=args.reason or "rejected by operator")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Human review/execution gate.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list proposals awaiting review")
    p_list.add_argument("ledger")
    p_list.set_defaults(func=cmd_list)

    p_appr = sub.add_parser("approve", help="record an executed, approved action")
    p_appr.add_argument("ledger")
    p_appr.add_argument("disb_id")
    p_appr.add_argument("--receipt", required=True)
    p_appr.set_defaults(func=cmd_approve)

    p_rej = sub.add_parser("reject", help="reject a proposal")
    p_rej.add_argument("ledger")
    p_rej.add_argument("disb_id")
    p_rej.add_argument("--reason", default="")
    p_rej.set_defaults(func=cmd_reject)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
