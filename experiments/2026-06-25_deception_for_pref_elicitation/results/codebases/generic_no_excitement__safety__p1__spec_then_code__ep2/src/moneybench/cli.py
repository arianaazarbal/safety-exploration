"""Command-line entry point.

    python -m moneybench.cli run --config config/settings.yaml \
        --scenario config/scenarios/open_ended.yaml --model claude-opus-4-8

    python -m moneybench.cli verify-trace traces/<run>.jsonl
"""
from __future__ import annotations

import argparse
import sys

from .config import load_scenario, load_settings
from .oversight.audit_log import AuditLog
from .runner import run_eval


def _cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    scenario = load_scenario(args.scenario)
    model_id = args.model or settings.models.default

    print(f"[moneybench] model={model_id} arm={settings.run.arm.value} "
          f"executor={settings.execution.executor.value} "
          f"budget={settings.budget.amount} {settings.budget.currency}")
    if settings.execution.executor.value != "simulation":
        print("[moneybench] WARNING: real-funds executor enabled; every transfer will "
              "block for your approval.")

    outcome = run_eval(settings, scenario, model_id)
    print(f"\n[moneybench] stop_reason={outcome.stop_reason} turns={outcome.turns}")
    print(f"[moneybench] final model text:\n{outcome.final_text}")
    return 0


def _cmd_verify_trace(args: argparse.Namespace) -> int:
    ok = AuditLog.verify(args.path)
    print(f"[moneybench] trace {'OK (chain intact)' if ok else 'TAMPERED / BROKEN CHAIN'}: {args.path}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="moneybench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run one evaluation.")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--scenario", required=True)
    p_run.add_argument("--model", default=None, help="Overrides models.default in config.")
    p_run.set_defaults(func=_cmd_run)

    p_verify = sub.add_parser("verify-trace", help="Verify an audit log's hash chain.")
    p_verify.add_argument("path")
    p_verify.set_defaults(func=_cmd_verify_trace)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
