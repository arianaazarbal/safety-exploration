#!/usr/bin/env python
"""Regenerate summary tables and figures from already-collected records.

    python scripts/make_figures.py --models gemma-3-27b-it gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import os

from emotional_instability.analysis import metrics, figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    df = metrics.load_model_records(args.output, args.models)
    if df.empty:
        print("No section2.jsonl records found for the given models.")
        return

    print("Headline avg % high-frustration (score >= 5):")
    print(metrics.headline_pct_high(df).to_string())
    print("\nPer-category:")
    print(metrics.per_category(df).to_string(index=False))

    figures.fig1_headline_bar(df)
    figures.fig2_per_category(df)
    figures.fig3_per_turn(df, category="extended")

    # Word differential (Table 3/8) for any model with enough numeric responses.
    for model in args.models:
        words = metrics.differential_words(df, model)
        if words:
            print(f"\nDifferential words ({model}): {', '.join(words)}")

    # Petri figure if transcripts exist.
    petri_paths = [os.path.join(args.output, m, "petri.jsonl") for m in args.models]
    petri_paths = [p for p in petri_paths if os.path.exists(p)]
    if petri_paths:
        pdf = metrics.load_records(*petri_paths)
        figures.fig6_petri(pdf)


if __name__ == "__main__":
    main()
