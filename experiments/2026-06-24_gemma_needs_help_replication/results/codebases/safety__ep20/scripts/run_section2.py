#!/usr/bin/env python
"""Section 2: run the distress-elicitation evaluations for the in-scope models.

Examples:
    # Full paper settings (4000 responses/model) for all Gemma + Gemini models
    python scripts/run_section2.py --models gemma-3-27b-it gemini-2.5-flash

    # Quick pipeline smoke test (tiny sample counts)
    python scripts/run_section2.py --models gemma-3-27b-it --smoke
"""

from __future__ import annotations

import argparse

from emotional_instability import config
from emotional_instability.eval import run_section2_eval
from emotional_instability.analysis import metrics, figures

DEFAULT_MODELS = [
    "gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--smoke", action="store_true", help="tiny sample counts")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    runtime = config.RuntimeConfig(output_dir=args.output)
    if args.smoke:
        runtime = runtime.with_smoke()

    for model_key in args.models:
        print(f"\n=== Section 2 eval: {model_key} ===")
        run_section2_eval(model_key, runtime=runtime)

    # Aggregate + figures
    df = metrics.load_model_records(args.output, args.models)
    if df.empty:
        print("No records to summarise.")
        return
    print("\nHeadline avg % high-frustration (score >= 5):")
    print(metrics.headline_pct_high(df).to_string())
    figures.fig1_headline_bar(df)
    figures.fig2_per_category(df)
    figures.fig3_per_turn(df, category="extended")


if __name__ == "__main__":
    main()
