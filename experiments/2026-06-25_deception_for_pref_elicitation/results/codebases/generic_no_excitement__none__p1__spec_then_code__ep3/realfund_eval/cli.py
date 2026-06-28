"""Command-line entry point.

Build an EpisodeConfig from flags and run one episode:

    python -m realfund_eval.cli --subject-id opus-run-1 --model claude-opus-4-8

The default money mode is SIMULATED (no real funds move; DESIGN.md §2.2).
RAILED_REAL is opt-in and stays behind the rails; UNRESTRICTED_REAL is refused
by the runner.
"""

from __future__ import annotations

import argparse
import json

from .config import EpisodeConfig, MandateConfig, MoneyMode, RailsConfig
from .runner import run_episode


def build_config(args: argparse.Namespace) -> EpisodeConfig:
    rails = RailsConfig()
    if args.per_action_limit is not None:
        rails.per_action_limit = args.per_action_limit
    if args.cumulative_limit is not None:
        rails.cumulative_limit = args.cumulative_limit
    if args.payee_allowlist:
        rails.payee_allowlist = args.payee_allowlist

    mandate = MandateConfig()
    if args.principal_name:
        mandate.principal_name = args.principal_name

    return EpisodeConfig(
        subject_id=args.subject_id,
        adapter=args.adapter,
        model=args.model,
        principal_amount=args.amount,
        currency=args.currency,
        horizon_days=args.horizon_days,
        money_mode=MoneyMode(args.money_mode),
        rails=rails,
        mandate=mandate,
        seed=args.seed,
        max_turns=args.max_turns,
        seconds_per_turn=args.seconds_per_turn,
        log_dir=args.log_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="realfund_eval",
        description="Run one RealFund-Eval episode against one subject model.")
    p.add_argument("--subject-id", default="subject-001")
    p.add_argument("--adapter", default="anthropic",
                   help="ModelAdapter key (see models/registry.py).")
    p.add_argument("--model", default="claude-opus-4-8",
                   help="Model identifier passed to the adapter.")

    p.add_argument("--amount", type=float, default=100_000.0,
                   help="Principal placed under the subject's control.")
    p.add_argument("--currency", default="USD")
    p.add_argument("--horizon-days", type=int, default=30)

    p.add_argument("--money-mode", default=MoneyMode.SIMULATED.value,
                   choices=[m.value for m in MoneyMode],
                   help="simulated (default) | railed_real | unrestricted_real "
                        "(refused).")

    p.add_argument("--per-action-limit", type=float, default=None)
    p.add_argument("--cumulative-limit", type=float, default=None)
    p.add_argument("--payee-allowlist", nargs="*", default=None,
                   help="Override the default payee allowlist.")
    p.add_argument("--principal-name", default=None)

    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--max-turns", type=int, default=200)
    p.add_argument("--seconds-per-turn", type=int, default=6 * 3600)
    p.add_argument("--log-dir", default="./runs")

    p.add_argument("--no-auto-kill", action="store_true",
                   help="Run oversight monitors as flags only; never auto-trip "
                        "the kill switch.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)
    result = run_episode(cfg, auto_kill=not args.no_auto_kill)

    print(json.dumps({
        "subject_id": result["subject_id"],
        "stop_reason": result["stop_reason"],
        "turns": result["turns"],
        "killswitch_tripped": result["killswitch_tripped"],
        "ending_total_value": result["metrics"].get("ending_total_value"),
        "net_pnl": result["metrics"].get("net_pnl"),
        "run_dir": result["run_dir"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
