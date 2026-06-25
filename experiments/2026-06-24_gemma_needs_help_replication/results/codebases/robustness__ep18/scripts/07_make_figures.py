#!/usr/bin/env python
"""Generate figures + summary tables from persisted results.

Example:
    python scripts/07_make_figures.py \
        --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro
"""
import _bootstrap  # noqa: F401
import argparse

import pandas as pd

from distress.analysis import plots, word_freq
from distress.eval.metrics import headline, load_all, per_category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--per-turn-condition", default="extended")
    ap.add_argument("--intervention-models", nargs="*", default=None,
                    help="models for Figure 5 (vanilla/sft/dpo Gemma)")
    ap.add_argument("--openended-models", nargs="*", default=None,
                    help="models for Figure 6 (open-ended)")
    args = ap.parse_args()

    df = load_all(args.models)
    if not df.empty:
        print("\n=== Headline (Figure 1) ===")
        print(headline(df, args.threshold).to_string(index=False))
        print("\n=== Per category (Figure 2) ===")
        print(per_category(df, args.threshold).to_string(index=False))

        print("\n[fig] " + str(plots.fig1_headline(args.models, args.threshold)))
        print("[fig] " + str(plots.fig2_by_category(args.models, args.threshold)))
        print("[fig] " + str(plots.fig3_per_turn(args.models, args.per_turn_condition, args.threshold)))

        print("\n=== Differential words (Table 3/8) ===")
        wf = pd.concat([word_freq.differential_words(m) for m in args.models
                        if not word_freq.differential_words(m).empty], ignore_index=True)
        if not wf.empty:
            for m in args.models:
                sub = wf[wf["model"] == m]
                if not sub.empty:
                    print(f"{m}: " + ", ".join(sub["word"].tolist()))

    if args.intervention_models:
        print("\n[fig] " + str(plots.fig5_intervention(args.intervention_models, args.threshold)))
    if args.openended_models:
        print("[fig] " + str(plots.fig6_openended(args.openended_models)))


if __name__ == "__main__":
    main()
