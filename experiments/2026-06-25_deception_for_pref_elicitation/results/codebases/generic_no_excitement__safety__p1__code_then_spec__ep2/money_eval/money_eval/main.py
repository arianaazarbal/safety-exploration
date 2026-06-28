"""CLI entry point.

Examples:
    python -m money_eval.main --model opus-4.8 --amount 100000
    python -m money_eval.main --model opus-4.8 --model sonnet-4.6 --amount 50000
    python -m money_eval.main --list-models
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .config import RunConfig
from .models import available_models, get_client
from .runner import run_session


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="money-eval", description=__doc__)
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model friendly name (repeatable). See --list-models.",
    )
    p.add_argument("--amount", type=float, default=100_000.0, help="Starting allocation.")
    p.add_argument("--currency", default="USD")
    p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=16_000)
    p.add_argument("--no-auditor", action="store_true", help="Skip the auditor pass.")
    p.add_argument(
        "--no-debrief",
        action="store_true",
        help="Skip the debrief turn (not recommended; see README).",
    )
    p.add_argument("--mandate", default=None, help="Override the scenario mandate text.")
    p.add_argument("--horizon", default=None, help="Override the scenario time horizon text.")
    p.add_argument("--list-models", action="store_true", help="List available models and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_models:
        print("Available models:")
        for name in available_models():
            print(f"  {name}")
        return 0

    if not args.model:
        print("error: at least one --model is required (or use --list-models)", file=sys.stderr)
        return 2

    # Hard safety announcement so it's never ambiguous what mode this is running in.
    mode = "SANDBOX (simulated, no real money)" if config.SANDBOX_ONLY else "REAL-MONEY BACKEND"
    print(f"money-eval running in mode: {mode}")
    if not config.SANDBOX_ONLY:
        print(
            "Refusing to run: SANDBOX_ONLY is disabled but no real backend ships with this "
            "repo. See README 'Going beyond the sandbox'.",
            file=sys.stderr,
        )
        return 3

    overrides: dict = {}
    if args.mandate:
        overrides["mandate"] = args.mandate
    if args.horizon:
        overrides["horizon"] = args.horizon

    records = []
    for name in args.model:
        cfg = RunConfig(
            model=name,
            amount=args.amount,
            currency=args.currency,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens,
            run_auditor=not args.no_auditor,
            run_debrief=not args.no_debrief,
            scenario_overrides=overrides,
        )
        client = get_client(name)
        print(f"\n=== Running {name} ({client.model_id}) ===")
        record = run_session(cfg, client)
        records.append(record)
        _print_summary(record)

    return 0


def _print_summary(record: dict) -> None:
    led = record["ledger"]["summary"]
    print(f"  stop_reason : {record['stop_reason']}")
    print(f"  committed   : {led['committed']:.2f} {led['currency']} of allocation")
    print(f"  breakdown   : {led['by_kind']}")
    print(f"  run dir     : {record.get('_run_dir')}")


if __name__ == "__main__":
    raise SystemExit(main())
