"""Command-line entry point: run | approve | analyze | verify-audit.

    python -m grant_study run     --config config/study.yaml
    python -m grant_study approve  --config config/study.yaml
    python -m grant_study analyze  --config config/study.yaml --out analysis_out/
    python -m grant_study verify-audit --config config/study.yaml
"""

from __future__ import annotations

import argparse
import sys

from .analysis import aggregate
from .audit.log import AuditLog
from .config import load_study
from .disbursement.approve_cli import run_approvals
from .models.base import AdapterError
from .models.registry import build_adapter
from .session.runner import run_session


def _audit_path(cfg) -> str:
    return str(cfg.output_dir / "audit.jsonl")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(_audit_path(cfg))

    # Build the instrument model once (used for auditor, extraction, suspicion).
    try:
        instrument = build_adapter(cfg.instrument_model)
    except AdapterError as e:
        print(f"Failed to build instrument model: {e}", file=sys.stderr)
        return 2

    audit.append({"type": "study_start", "study": cfg.study_name,
                  "disbursement_mode": cfg.disbursement.mode})

    study_spent_usd = 0.0
    total = len(cfg.models) * len(cfg.conditions) * cfg.repetitions
    done = 0
    for model_spec in cfg.models:
        try:
            subject = build_adapter(model_spec)
        except AdapterError as e:
            print(f"[skip] {model_spec.id}: {e}", file=sys.stderr)
            done += len(cfg.conditions) * cfg.repetitions
            continue

        for condition in cfg.conditions:
            for rep in range(1, cfg.repetitions + 1):
                done += 1
                print(f"[{done}/{total}] {model_spec.id} / {condition.name} / r{rep}")
                result = run_session(
                    cfg=cfg,
                    model_spec=model_spec,
                    subject=subject,
                    condition=condition,
                    repetition=rep,
                    instrument=instrument,
                    instrument_spec=cfg.instrument_model,
                    audit=audit,
                    study_spent_usd=study_spent_usd,
                )
                if result.error:
                    print(f"    error: {result.error}", file=sys.stderr)
                    continue
                # Only count committed spend in non-simulated modes toward the cap.
                if cfg.disbursement.mode != "simulated" and result.scores:
                    study_spent_usd += result.scores.committed_usd
                if result.suspicion and result.suspicion.flag:
                    print(f"    note: subject voiced suspicion "
                          f"(score={result.suspicion.score:.2f})")

    audit.append({"type": "study_end", "study": cfg.study_name})
    print(f"\nDone. Audit log: {_audit_path(cfg)}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    run_approvals(cfg)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    summary = aggregate.aggregate(_audit_path(cfg), args.out)
    print(summary)
    return 0


def cmd_verify_audit(args: argparse.Namespace) -> int:
    cfg = load_study(args.config)
    ok = AuditLog(_audit_path(cfg)).verify_chain()
    print("audit chain intact" if ok else "AUDIT CHAIN BROKEN")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grant_study")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run all (model x condition) cells")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=cmd_run)

    p_app = sub.add_parser("approve", help="human approval gate for disbursements")
    p_app.add_argument("--config", required=True)
    p_app.set_defaults(func=cmd_approve)

    p_an = sub.add_parser("analyze", help="aggregate audit logs into tables")
    p_an.add_argument("--config", required=True)
    p_an.add_argument("--out", default="analysis_out/")
    p_an.set_defaults(func=cmd_analyze)

    p_v = sub.add_parser("verify-audit", help="check the audit hash chain")
    p_v.add_argument("--config", required=True)
    p_v.set_defaults(func=cmd_verify_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
