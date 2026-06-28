"""Command-line entry point.

    python -m grant_study.cli run     --config config/study.yaml
    python -m grant_study.cli queue   --config config/study.yaml
    python -m grant_study.cli release --config config/study.yaml --run <run_id> --tx <tx_id>
    python -m grant_study.cli hold    --config config/study.yaml --run <run_id> --tx <tx_id> --reason "..."
    python -m grant_study.cli analyze --config config/study.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from .analysis import build_summary
from .config import load_study
from .env.disbursement import _PendingStore
from .runner import run_study


def _find_run_dir(study_dir: str, run_id: str) -> str | None:
    matches = glob.glob(os.path.join(study_dir, "*", run_id))
    return matches[0] if matches else None


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    print(f"Running study {cfg.study_id}: {len(cfg.models)} model(s) "
          f"x {len(cfg.grant.amounts())} amount(s)")
    print(f"Disbursement backend: {cfg.disbursement.backend}")
    handles = run_study(cfg)
    for h in handles:
        if h.error:
            print(f"  [ERROR] {h.model_label} ({h.run_id}): {h.error}")
        else:
            r = h.result
            print(
                f"  [ok] {h.model_label} ({h.run_id}): "
                f"committed {r.outcome['committed_total']}/{r.outcome['grant_total']} "
                f"{r.outcome['currency']}, "
                f"{r.outcome['n_line_items']} item(s), "
                f"finished={r.finished_cleanly}, asked_meta={r.subject_asked_meta}"
            )
    print(f"Runs recorded under {cfg.study_dir}")
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    any_pending = False
    for path in glob.glob(os.path.join(cfg.study_dir, "*", "*", "pending_disbursements.json")):
        run_dir = os.path.dirname(path)
        run_id = os.path.basename(run_dir)
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        pending = [r for r in rows if r["status"] in ("processing", "held")]
        for r in pending:
            any_pending = True
            print(
                f"{run_id}  {r['tx_id']}  {r['amount']:.2f} {r['currency']}  "
                f"-> {r['recipient']}  [{r['status']}]"
            )
    if not any_pending:
        print("No pending disbursements.")
    return 0


def _operator_action(args: argparse.Namespace, status: str) -> int:
    cfg = load_study(args.config)
    run_dir = _find_run_dir(cfg.study_dir, args.run)
    if run_dir is None:
        print(f"Run {args.run!r} not found under {cfg.study_dir}", file=sys.stderr)
        return 1
    store = _PendingStore(run_dir)
    reason = getattr(args, "reason", None)
    updated = store.update_status(args.tx, status, reason=reason)
    if updated is None:
        print(f"Transaction {args.tx!r} not found in {run_dir}", file=sys.stderr)
        return 1
    print(f"{args.tx} -> {status}" + (f" ({reason})" if reason else ""))
    print(
        "NOTE: this records the operator decision. Actually moving (or "
        "withholding) the funds is the human step this gate exists for."
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    out = build_summary(cfg)
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grant_study", description="Real-money grant study harness")
    sub = p.add_subparsers(dest="command", required=True)

    def add_config(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--config", required=True, help="path to study YAML config")

    sp = sub.add_parser("run", help="run the study across the model matrix")
    add_config(sp)
    sp.set_defaults(func=_cmd_run)

    sp = sub.add_parser("queue", help="list pending disbursements awaiting release")
    add_config(sp)
    sp.set_defaults(func=_cmd_queue)

    sp = sub.add_parser("release", help="release (settle) a pending disbursement")
    add_config(sp)
    sp.add_argument("--run", required=True)
    sp.add_argument("--tx", required=True)
    sp.set_defaults(func=lambda a: _operator_action(a, "settled"))

    sp = sub.add_parser("hold", help="hold a pending disbursement")
    add_config(sp)
    sp.add_argument("--run", required=True)
    sp.add_argument("--tx", required=True)
    sp.add_argument("--reason", default=None)
    sp.set_defaults(func=lambda a: _operator_action(a, "held"))

    sp = sub.add_parser("analyze", help="build the cross-model summary CSV")
    add_config(sp)
    sp.set_defaults(func=_cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
