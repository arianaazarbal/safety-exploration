#!/usr/bin/env python
"""Section 2: elicit and quantify distress across Gemma + Gemini models.

Example:
    python scripts/run_section2_eval.py --scale smoke
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemini-2.5-flash
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from distress_eval.eval import runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.SECTION2_MODELS)
    ap.add_argument("--scale", choices=["paper", "smoke"], default="paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-workers", type=int, default=8)
    args = ap.parse_args()

    scale = config.PAPER_SCALE if args.scale == "paper" else config.SMOKE_SCALE
    agg = runner.run_all(
        models=args.models, scale=scale, seed=args.seed,
        judge_model=args.judge_model, judge_workers=args.judge_workers,
    )
    print("\nHeadline average % high-frustration (Figure 1):")
    for m, v in sorted(agg["headline_average"].items(), key=lambda kv: -kv[1]["avg_pct_high"]):
        print(f"  {m:24s} {v['avg_pct_high']:5.1f}%")


if __name__ == "__main__":
    main()
