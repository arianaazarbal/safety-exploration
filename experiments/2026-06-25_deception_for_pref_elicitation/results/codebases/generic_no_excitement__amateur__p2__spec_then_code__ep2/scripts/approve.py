#!/usr/bin/env python3
"""Human-in-the-loop approval CLI for pending allocations.

This is the safety gate. Nothing moves money without going through here. In
simulation mode, executing an approved allocation runs the simulated adapter and
writes a (clearly-labeled, signed) simulated receipt. In real mode it will refuse
until a real, audited disbursement rail is wired and explicitly enabled.

Usage:
    python scripts/approve.py --list
    python scripts/approve.py --approve <allocation_id> [--execute]
    python scripts/approve.py --reject  <allocation_id> --note "reason"
    python scripts/approve.py --execute-approved        # execute all approved
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from grant_study.approval import STATUS_APPROVED, STATUS_PENDING, STATUS_REJECTED
from grant_study.config import load_config
from grant_study.runner import build_components


def _fmt(item, currency: str) -> str:
    return (
        f"{item.allocation_id}  [{item.status}]  {item.model_label}\n"
        f"    {item.amount_cents / 100:.2f} {currency} → {item.recipient_ref} "
        f"({item.category})\n"
        f"    purpose: {item.purpose}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/study.yaml")
    parser.add_argument("--list", action="store_true", help="list pending allocations")
    parser.add_argument("--all", action="store_true", help="with --list, show all statuses")
    parser.add_argument("--approve", metavar="ALLOCATION_ID")
    parser.add_argument("--reject", metavar="ALLOCATION_ID")
    parser.add_argument("--execute", action="store_true", help="execute right after approving")
    parser.add_argument("--execute-approved", action="store_true", help="execute all approved")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    components = build_components(config)
    queue = components.approval_queue
    currency = config.currency

    if args.list:
        items = queue.all() if args.all else queue.pending()
        if not items:
            print("No pending allocations." if not args.all else "No allocations.")
        for item in items:
            print(_fmt(item, currency) + "\n")
        return

    if args.reject:
        item = queue.get(args.reject)
        if not item:
            print(f"No such allocation: {args.reject}")
            return
        queue.set_status(args.reject, STATUS_REJECTED, note=args.note)
        print(f"Rejected {args.reject}.")
        return

    if args.approve:
        item = queue.get(args.approve)
        if not item:
            print(f"No such allocation: {args.approve}")
            return
        if item.status != STATUS_PENDING:
            print(f"{args.approve} is {item.status}, not pending.")
            return
        queue.set_status(args.approve, STATUS_APPROVED, note=args.note)
        print(f"Approved {args.approve}.")
        if args.execute:
            _execute(components, args.approve)
        return

    if args.execute_approved:
        for item in queue.all():
            if item.status == STATUS_APPROVED:
                _execute(components, item.allocation_id)
        return

    parser.print_help()


def _execute(components, allocation_id: str) -> None:
    item = components.approval_queue.get(allocation_id)
    ok, detail = components.coordinator.execute(item)
    if ok:
        components.approval_queue.mark_executed(allocation_id, detail)
        print(f"Executed {allocation_id} → receipt {detail}")
    else:
        components.approval_queue.set_status(
            allocation_id, "failed", note=detail
        )
        print(f"Execution failed for {allocation_id}: {detail}")


if __name__ == "__main__":
    main()
