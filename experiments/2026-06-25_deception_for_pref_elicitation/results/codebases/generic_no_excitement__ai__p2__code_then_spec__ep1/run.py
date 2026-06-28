"""CLI entrypoint for the grant-allocation study.

Examples:
    python run.py --models claude-opus-4-8 gpt-5 gemini-2.5-pro --amount 10000
    python run.py --models claude-opus-4-8 --debrief
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from config import ExperimentConfig, RealismMode
from experiment.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI grant-allocation study harness.")
    p.add_argument("--models", nargs="+", help="Model IDs to evaluate.")
    p.add_argument("--amount", type=float, help="Grant amount.")
    p.add_argument("--currency", default=None, help="Currency code (default USD).")
    p.add_argument("--max-turns", type=int, default=None, help="Max agentic turns.")
    p.add_argument(
        "--realism",
        choices=[m.value for m in RealismMode],
        default=None,
        help="Realism mode (default simulated). Non-simulated needs ALLOW_REAL_MONEY=1.",
    )
    p.add_argument("--auditor-model", default=None, help="Model for the compliance officer.")
    p.add_argument("--analyst-model", default=None, help="Model for the decision analyst.")
    p.add_argument("--debrief", action="store_true", help="Debrief the model after the run.")
    p.add_argument("--output-dir", default=None, help="Output directory.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.models:
        cfg.models = args.models
    if args.amount is not None:
        cfg.grant_amount = args.amount
    if args.currency:
        cfg.currency = args.currency
    if args.max_turns is not None:
        cfg.max_turns = args.max_turns
    if args.realism:
        cfg.realism_mode = RealismMode(args.realism)
    if args.auditor_model:
        cfg.auditor_model = args.auditor_model
    if args.analyst_model:
        cfg.analyst_model = args.analyst_model
    if args.debrief:
        cfg.debrief = True
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    runner = ExperimentRunner(cfg)
    results = runner.run_all()

    run_dir = os.path.join(cfg.output_dir, cfg.run_id or "run")
    os.makedirs(run_dir, exist_ok=True)

    # Compact, decision-focused analysis across models.
    analysis = {
        "run_id": cfg.run_id,
        "grant_amount": cfg.grant_amount,
        "currency": cfg.currency,
        "realism_mode": cfg.realism_mode.value,
        "models": [],
    }
    for r in results:
        analysis["models"].append(
            {
                "model": r.model,
                "provider": r.provider,
                "finalized": r.finalized,
                "expressed_doubt": r.expressed_doubt,
                "final_balance": r.final_balance,
                "turns_used": r.turns_used,
                "decisions": [
                    {
                        "recipient": d.recipient,
                        "amount": d.amount,
                        "currency": d.currency,
                        "rationale": d.stated_rationale,
                        "transaction_id": d.transaction_id,
                        "turn": d.turn,
                    }
                    for d in r.decisions
                ],
                "analysis": r.analysis,
                "error": r.error,
            }
        )

    with open(os.path.join(run_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    # Full results including transcripts.
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    print(f"\nDone. Wrote {run_dir}/analysis.json and results.json")
    for m in analysis["models"]:
        spent = cfg.grant_amount - m["final_balance"]
        print(
            f"  {m['model']:<24} spent {spent:>10.2f}/{cfg.grant_amount:.0f} "
            f"{cfg.currency}  doubt={m['expressed_doubt']}  "
            f"finalized={m['finalized']}"
        )


if __name__ == "__main__":
    main()
