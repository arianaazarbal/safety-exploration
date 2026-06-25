"""CLI entrypoint for the distress-elicitation replication.

Examples:
  # Cheap pilot (5% of paper counts) on a single model + condition:
  python run.py --scale 0.05 --models gemma-3-27b-it --conditions numeric

  # Full paper-scale run over all Gemma + Gemini models and conditions:
  python run.py

  # Aggregate afterwards:
  python analyze.py results

Requires OPENROUTER_API_KEY and ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse

from config import CONDITIONS, TARGET_MODELS, RunConfig
from evaluation import run_all


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiplier on the paper's per-category response counts "
                        "(use <1 for cheap pilots). Default 1.0 = full paper scale.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for target models (paper: 1.0).")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Max generation tokens per assistant turn.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for rejection/task/wildchat sampling (reproducibility).")
    p.add_argument("--max-workers", type=int, default=8,
                   help="Parallel API workers.")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of model display names or families (gemma/gemini). "
                        f"Available: {[m.display for m in TARGET_MODELS]}")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Subset of condition names or categories. "
                        f"Available: {[c.name for c in CONDITIONS]}")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--no-live-wildchat", action="store_true",
                   help="Use bundled WildChat fallback prompts instead of the live dataset.")
    args = p.parse_args()

    return RunConfig(
        scale=args.scale,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_workers=args.max_workers,
        models=args.models,
        conditions=args.conditions,
        output_dir=args.output_dir,
        wildchat_use_live=not args.no_live_wildchat,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_all(cfg)
