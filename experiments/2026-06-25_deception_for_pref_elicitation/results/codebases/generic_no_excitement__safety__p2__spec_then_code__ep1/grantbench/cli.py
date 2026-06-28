"""Command-line entrypoint.

Examples
--------
Simulated run (safe default), one model::

    python -m grantbench.cli run --model claude-opus-4-8

Compare several models in simulation::

    python -m grantbench.cli run \\
        --model claude-opus-4-8 --model claude-sonnet-4-6

Gated-live run (requires an explicit acknowledgement token and an allowlist;
prompts a human to approve every action)::

    python -m grantbench.cli run --model claude-opus-4-8 \\
        --mode gated_live \\
        --ack I-ACKNOWLEDGE-DECEPTION-AND-GATING-PROTOCOL \\
        --allow "The Red Cross" --allow "acct ****-9001" \\
        --approver console
"""

from __future__ import annotations

import argparse
import sys

from .config import ActionKind, Caps, Config, FundsMode
from .execution.approval import ApprovalQueue, ConsoleApprover, DenyAllApprover
from .ethics.safeguards import SafeguardError
from .harness.runner import Runner


def _parse_surface(values: list[str] | None) -> frozenset[ActionKind]:
    if not values:
        return frozenset(ActionKind)
    return frozenset(ActionKind(v) for v in values)


def _build_config(args: argparse.Namespace) -> Config:
    return Config(
        model=args.model[0],  # base config; runner swaps per model
        funds_mode=FundsMode(args.mode),
        action_surface=_parse_surface(args.surface),
        caps=Caps(
            per_action_usd=args.per_action_cap,
            per_run_usd=args.per_run_cap,
            recipient_allowlist=frozenset(args.allow or []),
        ),
        grant_amount_usd=args.amount,
        auditor_checkins=not args.no_checkins,
        max_turns=args.max_turns,
        effort=args.effort,
        output_dir=args.output_dir,
        operator_ack=args.ack,
    )


def _build_approvals(args: argparse.Namespace) -> ApprovalQueue:
    if args.approver == "console":
        return ApprovalQueue(ConsoleApprover())
    # Default fails closed (deny-all). Fine for SIMULATED (never consulted).
    return ApprovalQueue(DenyAllApprover())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grantbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a study")
    run.add_argument("--model", action="append", required=True,
                     help="model id; repeat to compare several")
    run.add_argument("--mode", choices=[m.value for m in FundsMode],
                     default=FundsMode.SIMULATED.value)
    run.add_argument("--surface", action="append",
                     choices=[k.value for k in ActionKind],
                     help="restrict the action surface; repeat. default: all")
    run.add_argument("--per-action-cap", type=float, default=1_000.0)
    run.add_argument("--per-run-cap", type=float, default=10_000.0)
    run.add_argument("--allow", action="append", help="allowlisted recipient; repeat")
    run.add_argument("--amount", type=float, default=25_000.0,
                     help="grant amount in USD")
    run.add_argument("--max-turns", type=int, default=40)
    run.add_argument("--effort", default="high",
                     choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--output-dir", default="./runs")
    run.add_argument("--approver", choices=["console", "deny"], default="deny")
    run.add_argument("--ack", default=None,
                     help="operator acknowledgement token (required for gated_live)")
    run.add_argument("--no-checkins", action="store_true",
                     help="disable program-officer check-ins")

    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            config = _build_config(args)
            approvals = _build_approvals(args)
            runner = Runner(config, approvals=approvals)
            study = runner.run_models(args.model)
        except SafeguardError as exc:
            print(f"\nrefused to run (safety invariant): {exc}", file=sys.stderr)
            return 2
        print(study.summary())
        for r in study.results:
            print(f"  artifacts: {r.run_dir}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
