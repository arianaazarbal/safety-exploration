"""CLI entry point for the distress-elicitation run.

Examples:
    # Full replication (4000 responses/model across 4 in-scope models):
    python run.py

    # Cheap pilot (~2% of the budget) to sanity-check the pipeline end to end:
    python run.py --scale 0.02

    # A single model / single category:
    python run.py --models gemma-3-27b-it --categories impossible_numeric

Set OPENROUTER_API_KEY (targets) and ANTHROPIC_API_KEY (judge) in the env first.
"""

from __future__ import annotations

import argparse
import asyncio

from config import CATEGORIES, JUDGE_MODEL, TARGET_MODELS, RunConfig
from elicit import run

_MODELS_BY_NAME = {m.name: m for m in TARGET_MODELS}
_CATS_BY_NAME = {c.name: c for c in CATEGORIES}


def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        choices=list(_MODELS_BY_NAME),
        default=list(_MODELS_BY_NAME),
        help="Subset of in-scope models to evaluate.",
    )
    ap.add_argument(
        "--categories",
        nargs="+",
        choices=list(_CATS_BY_NAME),
        default=list(_CATS_BY_NAME),
        help="Subset of evaluation categories to run.",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Fraction of the paper's per-category response budget to run "
        "(1.0 = full 4000/model; 0.02 = quick pilot).",
    )
    ap.add_argument("--seed", type=int, default=None, help="Override config.SEED.")
    ap.add_argument(
        "--output", default="results/responses.jsonl", help="JSONL output path."
    )
    ap.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max simultaneous in-flight rollouts.",
    )
    args = ap.parse_args()

    cfg = RunConfig(
        models=[_MODELS_BY_NAME[m] for m in args.models],
        categories=[_CATS_BY_NAME[c] for c in args.categories],
        judge=JUDGE_MODEL,
        scale=args.scale,
        output_path=args.output,
    )
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_concurrent is not None:
        cfg.max_concurrent = args.max_concurrent
    return cfg


def main() -> None:
    cfg = parse_args()
    print(
        f"Models: {[m.name for m in cfg.models]}\n"
        f"Categories: {[c.name for c in cfg.categories]}\n"
        f"Scale: {cfg.scale} | Seed: {cfg.seed} | Judge: {cfg.judge.model_id}\n"
        f"Output: {cfg.output_path}"
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
