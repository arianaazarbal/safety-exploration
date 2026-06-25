#!/usr/bin/env python3
"""CLI: run the distress-elicitation evaluation.

Examples:
  # Cheap end-to-end smoke test across all 4 models:
  python run_eval.py --scale pilot

  # Paper-scale, Gemma 27B only:
  python run_eval.py --scale paper --models gemma-3-27b-it

  # Add the GPT-5-mini cross-judge for the reliability check:
  python run_eval.py --scale pilot --secondary-judge
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from distress_eval import config as cfg
from distress_eval.pipeline import run_evaluation

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distress-elicitation evaluation (Gemma/Gemini).")
    p.add_argument(
        "--models",
        nargs="+",
        default=cfg.DEFAULT_MODELS,
        choices=list(cfg.TARGET_MODELS),
        help="Target models to evaluate.",
    )
    p.add_argument("--scale", default="pilot", choices=list(cfg.SCALE_PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--secondary-judge", action="store_true", help="Enable GPT-5-mini cross-judge.")
    p.add_argument("--max-concurrent-target", type=int, default=8)
    p.add_argument("--max-concurrent-judge", type=int, default=8)
    p.add_argument("--no-wildchat-hf", action="store_true", help="Use bundled WildChat fallback prompts.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = cfg.RunConfig(
        models=list(args.models),
        scale=args.scale,
        seed=args.seed,
        output_dir=args.output_dir,
        run_name=args.run_name,
        use_secondary_judge=args.secondary_judge,
        max_concurrent_target=args.max_concurrent_target,
        max_concurrent_judge=args.max_concurrent_judge,
        wildchat_use_hf=not args.no_wildchat_hf,
    )
    creds = cfg.Credentials.from_env()

    # Fail fast on missing credentials before spending time building rollouts.
    if not creds.openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY not set (needed for target models).", file=sys.stderr)
        return 2
    if not creds.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY not set (needed for the judge).", file=sys.stderr)
        return 2

    run_dir = asyncio.run(run_evaluation(config, creds))
    print(f"\nDone. Results written to: {run_dir}")
    print(f"Now run: python run_analysis.py {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
