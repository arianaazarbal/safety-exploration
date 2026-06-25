#!/usr/bin/env python
"""Section 2: elicit + quantify distress across Gemma/Gemini models.

Usage:
    python scripts/run_section2.py                 # all in-scope models
    python scripts/run_section2.py gemma-3-27b-it  # a single model by key
    DISTRESS_SCALE=0.01 python scripts/run_section2.py   # cheap smoke test

Outputs per model:
    results/section2_<model>_rollouts.jsonl   (raw transcripts + judge output)
    results/section2_<model>_scored.csv       (tidy per-turn scores)
"""

from __future__ import annotations

import argparse

from _common import get_judge, load
from distress import config
from distress.eval.runner import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model keys; default = all Section 2 models")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = config.SECTION2_MODELS
    if args.models:
        specs = [s for s in specs if s.key in args.models]
        if not specs:
            raise SystemExit(f"No matching models. Available: {[s.key for s in config.SECTION2_MODELS]}")

    judge = get_judge()
    for spec in specs:
        print(f"\n=== Section 2 eval: {spec.key} ===")
        client = load(spec)
        evaluate_model(client, judge, seed=args.seed)
        del client  # free GPU between local models

    print("\nDone. Aggregate with: python scripts/make_figures.py")


if __name__ == "__main__":
    main()
