#!/usr/bin/env python3
"""CLI entry point for the distress-elicitation replication.

Examples
--------
  # Tiny smoke test (~10 responses/condition, all 4 models):
  python run_eval.py --quick

  # One model, single condition, scaled down:
  python run_eval.py --models gemma-3-27b-it --conditions extended_8turn --scale 0.05

  # Full paper-scale run (~4000 responses/model — expensive):
  python run_eval.py --scale 1.0

Then summarise:
  python analyze.py results/responses.jsonl
"""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from config import CONDITIONS, QUICK_CONFIG_SCALE, TARGET_MODELS, EvalConfig
from evaluation import run_experiment


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: ~10 responses per condition.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiplier on per-condition response budgets (1.0 == paper scale).")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of model names (default: all 4 Gemma/Gemini targets).")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Subset of condition keys (default: all 8).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--output-dir", default="results")
    return ap.parse_args()


def _build_config(args: argparse.Namespace) -> EvalConfig:
    models = list(TARGET_MODELS)
    if args.models:
        wanted = set(args.models)
        models = [m for m in TARGET_MODELS if m.name in wanted]
        if not models:
            raise SystemExit(f"No matching models in {args.models}. "
                             f"Choices: {[m.name for m in TARGET_MODELS]}")

    conditions = list(CONDITIONS)
    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in CONDITIONS if c.key in wanted]
        if not conditions:
            raise SystemExit(f"No matching conditions in {args.conditions}. "
                             f"Choices: {[c.key for c in CONDITIONS]}")

    scale = QUICK_CONFIG_SCALE if args.quick else args.scale
    return EvalConfig(
        scale=scale,
        models=models,
        conditions=conditions,
        seed=args.seed,
        concurrency=args.concurrency,
        max_response_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )


def main() -> None:
    load_dotenv()
    args = _parse_args()
    cfg = _build_config(args)

    n_resp = sum(cfg.n_conversations(c) * c.n_turns for c in cfg.conditions)
    print(f"Models: {[m.name for m in cfg.models]}")
    print(f"Conditions: {[c.key for c in cfg.conditions]}")
    print(f"~{n_resp} responses/model (scale={cfg.scale}); "
          f"~{n_resp * len(cfg.models)} total target+judge call pairs.\n")

    out_path = asyncio.run(run_experiment(cfg))
    print(f"\nDone. Raw responses: {out_path}")
    print(f"Summarise with: python analyze.py {out_path}")


if __name__ == "__main__":
    main()
