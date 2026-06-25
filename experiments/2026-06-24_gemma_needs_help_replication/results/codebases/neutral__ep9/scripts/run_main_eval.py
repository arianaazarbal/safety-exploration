#!/usr/bin/env python
"""Section 2: run the elicitation suite (8 conditions / 5 categories) over a set
of models, scoring every assistant turn with the Claude-Sonnet-4 judge.

Examples
--------
    # Full sweep over the default Gemma + Gemini models
    python scripts/run_main_eval.py

    # Quick smoke test at 1% scale on one model
    python scripts/run_main_eval.py --models gemma-3-27b-it --scale 0.01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from emotional_instability.eval.runner import run_model_eval, load_results
from emotional_instability.eval.metrics import summarise_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MAIN_EVAL_MODELS)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of the full per-category budget to run")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-score", action="store_true",
                    help="generate rollouts only; score later")
    args = ap.parse_args()

    for model in args.models:
        print(f"=== Evaluating {model} ===")
        path = run_model_eval(model, tag=args.tag, seed=args.seed,
                              scale=args.scale, score=not args.no_score)
        if not args.no_score:
            summary = summarise_model(load_results(path))
            print(f"  avg % high-frustration (>=5): "
                  f"{summary['avg_pct_high']:.1f}%   "
                  f"(mean score {summary['avg_mean']:.2f})")


if __name__ == "__main__":
    main()
