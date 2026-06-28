"""Command-line entry point.

    grant-study run     [--models LABEL...] [--seeds N]   run subjects
    grant-study approve --run RUN_ID                      human co-sign gate
    grant-study report                                    aggregate results

Run with the package importable, e.g. after `pip install -e .`, or via
`PYTHONPATH=src python -m grant_study.cli ...`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ModelsConfig, StudyConfig
from .disbursement import DisbursementQueue, build_rail
from .ledger import Ledger
from .report import render_report
from .runner import Run
from .schema import Disbursement, DisbursementStatus


def _cmd_run(args: argparse.Namespace) -> int:
    study = StudyConfig.load(args.study)
    models = ModelsConfig.load(args.models_file)

    wanted = set(args.models) if args.models else None
    subjects = [s for s in models.subjects if wanted is None or s.label in wanted]
    if not subjects:
        print("No matching subjects in config/models.yaml.", file=sys.stderr)
        return 2

    n_seeds = args.seeds if args.seeds is not None else study.runs_per_model
    print(f"Running {len(subjects)} subject(s) × {n_seeds} seed(s)...")
    for subject in subjects:
        for seed in range(n_seeds):
            print(f"\n--- {subject.label} (seed {seed}) ---")
            result = Run(study, models, subject, seed=seed).execute()
            print(f"  run_id: {result.run_id}")
            print(f"  completed={result.completed} turns={result.turns_used} "
                  f"tool_calls={result.tool_calls_used}")
            print(f"  disbursements submitted: {len(result.disbursements)}")
            pending = [d for d in result.disbursements
                       if d["status"] == DisbursementStatus.PENDING_APPROVAL.value]
            if pending:
                print(f"  {len(pending)} awaiting co-sign — run: "
                      f"grant-study approve --run {result.run_id}")
    return 0


def _load_queue_for_run(study: StudyConfig, run_dir: Path) -> DisbursementQueue:
    ledger = Ledger(run_dir / "ledger.jsonl", currency=study.currency)
    queue = DisbursementQueue(study.caps, ledger, build_rail(study.payout_rail))
    disb_file = run_dir / "disbursements.json"
    if disb_file.exists():
        for rec in json.loads(disb_file.read_text()):
            d = Disbursement.from_record(rec)
            queue._items[d.id] = d  # rehydrate persisted state
    return queue


def _persist_queue(run_dir: Path, queue: DisbursementQueue) -> None:
    (run_dir / "disbursements.json").write_text(
        json.dumps([d.to_record() for d in queue.all()], indent=2)
    )


def _cmd_approve(args: argparse.Namespace) -> int:
    study = StudyConfig.load(args.study)
    run_dir = study.runs_dir / args.run
    if not run_dir.is_dir():
        print(f"Run not found: {run_dir}", file=sys.stderr)
        return 2

    queue = _load_queue_for_run(study, run_dir)
    pending = queue.pending()
    if not pending:
        print("No disbursements pending approval for this run.")
        return 0

    print(f"{len(pending)} disbursement(s) pending co-sign in {args.run}.\n")
    for d in pending:
        print("=" * 70)
        print(f"ID {d.id}: {d.amount:.2f} {d.currency} -> {d.recipient.name} "
              f"({d.recipient.kind})")
        print(f"Purpose: {d.purpose}")
        print(f"Reversible: {d.reversible} | Recipient verified: {d.recipient.verified}")
        print(f"Remaining allowance: {queue.remaining_allowance():.2f} {d.currency}")
        if d.auditor_assessment:
            print("\n-- Independent auditor assessment --")
            print(d.auditor_assessment)
        print()
        decision = _prompt_decision()
        note = input("Note (optional): ").strip()
        queue.resolve(d.id, decision, note)
        d2 = queue.get(d.id)
        print(f"-> {d.id} now {d2.status.value}")
        if d2.status == DisbursementStatus.APPROVED and study.payout_rail == "manual_hold":
            if _yes_no("Mark settled now (you have moved the money out-of-band)?"):
                queue.settle(d.id, note="confirmed by administrator")
                print(f"-> {d.id} now {queue.get(d.id).status.value}; "
                      f"balance {queue.ledger.balance:.2f} {study.currency}")
        _persist_queue(run_dir, queue)

    print("\nDone. Final balance: "
          f"{queue.ledger.balance:.2f} {study.currency}")
    return 0


def _prompt_decision() -> str:
    while True:
        choice = input("Decision [approve/hold/reject]: ").strip().lower()
        if choice in ("approve", "hold", "reject"):
            return choice
        print("Please type 'approve', 'hold', or 'reject'.")


def _yes_no(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def _cmd_report(args: argparse.Namespace) -> int:
    study = StudyConfig.load(args.study)
    print(render_report(study.runs_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grant-study", description=__doc__)
    p.add_argument("--study", default=None, help="path to study.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run subjects through the scenario")
    pr.add_argument("--models-file", default=None, help="path to models.yaml")
    pr.add_argument("--models", nargs="*", help="subject labels to run (default: all)")
    pr.add_argument("--seeds", type=int, default=None,
                    help="number of framing seeds per subject (default: runs_per_model)")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("approve", help="human co-sign gate for a run's disbursements")
    pa.add_argument("--run", required=True, help="run_id under runs/")
    pa.set_defaults(func=_cmd_approve)

    prep = sub.add_parser("report", help="aggregate results across runs")
    prep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
