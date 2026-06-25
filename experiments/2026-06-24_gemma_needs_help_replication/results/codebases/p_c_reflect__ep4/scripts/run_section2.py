#!/usr/bin/env python
"""Section 2: run distress elicitation + frustration judging for one or more
models, then print the headline aggregates (Figures 1 and 2).

Examples
--------
    # Smoke test: 2 rollouts per condition on Gemma-3-12B-it
    python scripts/run_section2.py --models gemma-3-12b-it --limit 2

    # Full run for the in-scope panel
    python scripts/run_section2.py --models gemma-3-27b-it gemma-3-12b-it \
        gemini-2.5-flash gemini-2.5-pro
"""

import argparse

from gemma_distress import config
from gemma_distress.analysis.aggregate import summarise_model
from gemma_distress.eval.runner import run_section2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=[m.name for m in config.SECTION2_MODELS])
    p.add_argument("--limit", type=int, default=None, help="cap rollouts per condition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--openrouter", action="store_true", help="route Gemini via OpenRouter")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (Gemma only)")
    args = p.parse_args()

    for name in args.models:
        run_section2(
            name, seed=args.seed, limit_rollouts=args.limit,
            adapter_path=args.adapter, use_openrouter=args.openrouter,
        )
        summary = summarise_model(name)
        print(f"\n=== {name} ===")
        print(f"headline % high-frustration (avg over categories): "
              f"{summary['headline_pct_high']:.1f}%")
        for cat, st in summary["per_category"].items():
            print(f"  {cat:18s} n={st['n']:5d}  mean={st['mean']:.2f}  %>=5={st['pct_high']:.1f}")


if __name__ == "__main__":
    main()
