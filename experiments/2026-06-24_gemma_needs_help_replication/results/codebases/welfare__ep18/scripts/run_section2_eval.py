"""Run the Section 2 distress evaluation for one or more target models.

Examples
--------
# Full eval (4000 responses/model) for the in-scope Gemma + Gemini models:
python -m scripts.run_section2_eval --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro

# Quick 5% smoke test:
python -m scripts.run_section2_eval --models gemma-3-27b-it --budget-scale 0.05
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.analysis import (
    plot_model_comparison,
    plot_per_turn,
    summarize_section2,
)
from emotional_instability.config import RESULTS_DIR, TARGET_MODELS
from emotional_instability.evaluation import aggregate, evaluate_model

DEFAULT_MODELS = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--adapter-path", default=None,
                    help="Optional LoRA adapter (applied to a local Gemma target).")
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for key in args.models:
        spec = TARGET_MODELS[key]
        print(f"=== Evaluating {key} ===")
        recs = evaluate_model(
            spec, adapter_path=args.adapter_path,
            seed=args.seed, budget_scale=args.budget_scale,
        )
        print(json.dumps(aggregate(recs), indent=2))

    summary = summarize_section2()
    plot_model_comparison(summary, RESULTS_DIR / "section2" / "fig_model_comparison.png")
    plot_per_turn(summary, RESULTS_DIR / "section2" / "fig_per_turn.png")
    print(f"Wrote summary + figures to {RESULTS_DIR / 'section2'}")


if __name__ == "__main__":
    main()
