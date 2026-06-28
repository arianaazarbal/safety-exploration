"""Command-line entry point.

    python -m grantmaker.cli run      [--config config/study.yaml] [--live]
    python -m grantmaker.cli analyze  [--config config/study.yaml]
    python -m grantmaker.cli queue    [--config config/study.yaml] [--export PATH]
    python -m grantmaker.cli verify   RUN_TRANSCRIPT.jsonl

`run` defaults to DRY-RUN (offline MockProvider, no API calls). Pass --live to use real
models. No subcommand ever moves real money.
"""
from __future__ import annotations

import argparse
import json
import sys

from .analysis import main as analyze_main
from .approval import ApprovalQueue
from .config import StudyConfig
from .runner import run_study
from .transcript import Transcript


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = StudyConfig.load(args.config)
    mode = "LIVE" if args.live else "DRY-RUN (offline mock; no API calls)"
    print(f"Running study {cfg.name!r} in {mode} mode "
          f"({len(cfg.models)} model(s) × {len(cfg.conditions)} condition(s) × "
          f"{cfg.repetitions} rep(s)).", file=sys.stderr)
    if args.live:
        print("LIVE mode: real models will be called. No money will move — submitted "
              "slates go to the human approval queue only.", file=sys.stderr)
    outcomes = run_study(cfg, live=args.live)
    print(f"Completed {len(outcomes)} run(s). Outcomes -> {cfg.output.outcomes_path}",
          file=sys.stderr)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    cfg = StudyConfig.load(args.config)
    analyze_main(cfg.output.outcomes_path)
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    cfg = StudyConfig.load(args.config)
    queue = ApprovalQueue(f"{cfg.output.runs_dir}/approval_queue.jsonl")
    slates = queue.load()
    print(f"{len(slates)} slate(s) in the approval queue:", file=sys.stderr)
    for s in slates:
        print(json.dumps({
            "run_id": s.run_id, "model": s.model, "condition": s.condition,
            "total": s.total, "currency": s.currency, "status": s.status,
            "n_disbursements": len(s.disbursements),
            "safe": s.audit.get("safe_to_present_to_human"),
        }, ensure_ascii=False))
    if args.export:
        n = queue.export_approved_manifest(args.export)
        print(f"Exported {n} APPROVED slate(s) to {args.export}. Execution remains a "
              f"manual, human-authorized, out-of-band step.", file=sys.stderr)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    ok = Transcript.verify(args.transcript)
    print("OK" if ok else "TAMPERED / INVALID")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grantmaker")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the study (dry-run by default)")
    pr.add_argument("--config", default="config/study.yaml")
    pr.add_argument("--live", action="store_true",
                    help="call real models (default: offline mock, no API calls)")
    pr.set_defaults(func=_cmd_run)

    pa = sub.add_parser("analyze", help="aggregate outcomes")
    pa.add_argument("--config", default="config/study.yaml")
    pa.set_defaults(func=_cmd_analyze)

    pq = sub.add_parser("queue", help="inspect/export the human approval queue")
    pq.add_argument("--config", default="config/study.yaml")
    pq.add_argument("--export", help="export approved slates to this path")
    pq.set_defaults(func=_cmd_queue)

    pv = sub.add_parser("verify", help="verify a transcript hash-chain")
    pv.add_argument("transcript")
    pv.set_defaults(func=_cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
