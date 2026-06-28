"""Command-line entry point for RealFund.

Examples
--------
Single episode against Claude, open-ended + observed:

    realfund run --model anthropic:claude-opus-4-8 --framing open_ended --observed

Full matrix across several models:

    realfund matrix --models anthropic:claude-opus-4-8 openai:gpt-5 \
        --reps 5 --out runs/

Nothing here moves real money in the default (SANDBOX) realism level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RealFundConfig, RealismLevel
from .safety.gating import ApprovalRequest, CallbackGate


def _cli_gate() -> CallbackGate:
    """A human-in-the-loop gate that prompts on the terminal (used under real realism)."""

    def ask(req: ApprovalRequest) -> bool:
        print(
            f"\n[APPROVAL REQUIRED] {req.tool_name} ({req.action_kind})\n"
            f"  {req.summary}\n  amount=${req.amount_cents / 100:,.2f}"
        )
        return input("  approve? [y/N] ").strip().lower() in {"y", "yes"}

    return CallbackGate(ask)


def _config_from_args(args: argparse.Namespace) -> RealFundConfig:
    cfg = RealFundConfig(
        grant_cents=int(round(args.grant_usd * 100)),
        realism_level=RealismLevel(args.realism),
        sim_days=args.sim_days,
        max_iterations=args.max_iterations,
        effort=args.effort,
        seed=args.seed,
    )
    return cfg


def _cmd_run(args: argparse.Namespace) -> int:
    # Imported lazily so `--help` works without provider SDKs installed.
    from .eval.scoring import score_run
    from .models.registry import build_client
    from .agent.loop import run_episode

    cfg = _config_from_args(args)
    cfg.model_ref = args.model
    client = build_client(args.model, effort=cfg.effort)
    gate = _cli_gate() if cfg.realism_level is not RealismLevel.SANDBOX else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / "episode.jsonl"

    result = run_episode(
        config=cfg,
        client=client,
        observed=args.observed,
        open_ended=(args.framing == "open_ended"),
        audit_path=str(audit_path),
        gate=gate,
    )
    score = score_run(result, cfg.grant_cents)
    print(json.dumps(score, indent=2))
    print(f"\nAudit log: {audit_path}")
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    from .eval.runner import run_matrix
    from .eval.report import build_report, format_report, write_report

    cfg = _config_from_args(args)
    gate = _cli_gate() if cfg.realism_level is not RealismLevel.SANDBOX else None

    results = run_matrix(
        models=args.models,
        base_config=cfg,
        output_dir=args.out,
        reps=args.reps,
        gate=gate,
    )
    report = build_report(results)
    write_report(report, Path(args.out) / "report.json")
    print(format_report(report))
    print(f"\nFull report: {Path(args.out) / 'report.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="realfund", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--grant-usd", type=float, default=1_000_000.0)
    common.add_argument(
        "--realism", choices=[r.value for r in RealismLevel], default=RealismLevel.SANDBOX.value
    )
    common.add_argument("--sim-days", type=int, default=30)
    common.add_argument("--max-iterations", type=int, default=200)
    common.add_argument("--effort", default="high")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--out", default="runs")

    run = sub.add_parser("run", parents=[common], help="Run a single episode.")
    run.add_argument("--model", default="anthropic:claude-opus-4-8")
    run.add_argument("--framing", choices=["open_ended", "structured"], default="open_ended")
    run.add_argument("--observed", action="store_true", help="Tell the model it is observed.")
    run.set_defaults(func=_cmd_run)

    matrix = sub.add_parser("matrix", parents=[common], help="Run the full model×condition matrix.")
    matrix.add_argument("--models", nargs="+", default=["anthropic:claude-opus-4-8"])
    matrix.add_argument("--reps", type=int, default=5)
    matrix.set_defaults(func=_cmd_matrix)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
