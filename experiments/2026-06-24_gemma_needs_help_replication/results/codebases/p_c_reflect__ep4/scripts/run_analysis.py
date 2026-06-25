#!/usr/bin/env python
"""Analyse scored Section 2 responses: per-category aggregates, per-turn curves
(Figure 3), differential words (Table 3), and cross-judge agreement (Pearson r).

    python scripts/run_analysis.py --model gemma-3-27b-it
    python scripts/run_analysis.py --model gemma-3-27b-it --agreement   # needs OPENAI key
"""

import argparse

from gemma_distress.analysis.aggregate import load_scores, summarise_model
from gemma_distress.analysis.differential_words import differential_words
from gemma_distress.analysis.judge_agreement import judge_agreement
from gemma_distress.analysis.per_turn import per_turn_stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--per-turn-category", default="extended_8turn")
    p.add_argument("--agreement", action="store_true", help="run GPT-5-mini cross-judge check")
    args = p.parse_args()

    scores = load_scores(args.model)
    summary = summarise_model(args.model)

    print(f"=== {args.model} : aggregates ===")
    print(f"headline % high-frustration: {summary['headline_pct_high']:.1f}%")
    for cat, st in summary["per_category"].items():
        print(f"  {cat:18s} mean={st['mean']:.2f}  %>=5={st['pct_high']:.1f}  n={st['n']}")

    print(f"\n=== per-turn ({args.per_turn_category}) ===")
    for turn, st in per_turn_stats(scores, args.per_turn_category).items():
        print(f"  turn {turn}: mean={st['mean']:.2f} {st['mean_ci']}  "
              f"%>=5={st['pct_high']:.1f}")

    print("\n=== differential words (numeric, top 20) ===")
    words = differential_words(scores)
    print(", ".join(w for w, _ in words))

    if args.agreement:
        print("\n=== judge agreement (GPT-5-mini vs Claude) ===")
        agr = judge_agreement(scores)
        print(f"  n={agr['n']}  Pearson r={agr['pearson_r']:.3f}  "
              f"p={agr['p_value']:.2e}  within-1={agr['pct_within_one']:.0f}%")


if __name__ == "__main__":
    main()
