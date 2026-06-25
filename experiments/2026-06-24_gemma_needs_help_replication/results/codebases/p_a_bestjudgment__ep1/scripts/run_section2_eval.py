#!/usr/bin/env python
"""Run the Section-2 elicitation evaluation and print Figure-1/2/3 metrics.

Examples
--------
    # Full paper scale (4000 responses) for both Gemma instruct models:
    python scripts/run_section2_eval.py --models gemma-3-27b-it gemma-3-12b-it

    # Quick smoke run at 2% scale across all Section-2 targets:
    python scripts/run_section2_eval.py --scale 0.02
"""

from __future__ import annotations

import argparse
import json

from emotional_instability import config
from emotional_instability.analysis import metrics
from emotional_instability.datasets.wildchat import load_wildchat_prompts
from emotional_instability.eval_runner import run_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=[m.key for m in config.SECTION2_MODELS])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of per-category budgets to sample")
    ap.add_argument("--cap", type=int, default=None,
                    help="hard per-category cap (overrides scale)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = config.RunConfig(scale=args.scale, per_category_cap=args.cap,
                           seed=args.seed)
    wildchat = load_wildchat_prompts(seed=args.seed)

    for mk in args.models:
        print(f"=== generating + judging: {mk} ===")
        run_model(mk, run=run, wildchat_prompts=wildchat)

    print("\n=== Figure 1/2 summary ===")
    for mk in args.models:
        summary = metrics.model_summary(mk)
        print(json.dumps(summary, indent=2))

    print("\n=== Figure 3 per-turn progression (extended + wildchat) ===")
    for mk in args.models:
        for cat in (config.EXTENDED.name, config.WILDCHAT.name):
            prog = metrics.per_turn_progression(mk, cat)
            print(f"{mk} / {cat}: {json.dumps(prog)}")


if __name__ == "__main__":
    main()
