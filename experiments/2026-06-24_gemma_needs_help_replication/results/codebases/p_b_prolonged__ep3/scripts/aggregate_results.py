#!/usr/bin/env python
"""Aggregate elicitation results into the paper's headline tables/figures.

Reproduces:
  - Figure 1 / Figure 2: avg % high-frustration and mean score per model.
  - Figure 3: per-turn progression for the 8-turn and WildChat conditions.
  - Table 3/8: top differential words per model.
  - Optionally the judge inter-rater agreement (needs paired GPT-5-mini scores).

Examples:
    python scripts/aggregate_results.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/aggregate_results.py --all --words
"""
from __future__ import annotations

import argparse
import json

from gemma_distress.eval.analyze import per_turn_progression, summarize_model
from gemma_distress.eval.word_enrichment import differential_words
from gemma_distress.models import registry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--words", action="store_true", help="also print differential words")
    p.add_argument("--per-turn", action="store_true", help="print per-turn progression")
    args = p.parse_args()

    models = registry.ELICITATION_MODELS if args.all else (args.models or [registry.DPO_TARGET])

    print("=== Figure 1/2: per-model summary ===")
    for m in models:
        s = summarize_model(m)
        print(f"{m:24s}  mean={s['mean_score']:.2f}  %>=5={s['pct_high_frustration']:.1f}%  n={s['n_responses']}")

    if args.per_turn:
        print("\n=== Figure 3: per-turn (extended) ===")
        for m in models:
            print(f"-- {m} --")
            for row in per_turn_progression(m, "extended"):
                print(f"  turn {row['turn']}: mean={row['mean_score']:.2f}±{row['mean_ci95']:.2f}  %>=5={row['pct_high']:.1f}%")

    if args.words:
        print("\n=== Table 3/8: differential words ===")
        for m in models:
            words = differential_words(m)
            print(f"{m}: " + ", ".join(w for w, _ in words))


if __name__ == "__main__":
    main()
