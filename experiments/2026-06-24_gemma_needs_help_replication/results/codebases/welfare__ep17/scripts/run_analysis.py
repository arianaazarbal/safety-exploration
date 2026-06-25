#!/usr/bin/env python3
"""Aggregate scored responses into the paper's headline tables and figures.

Produces:
  - outputs/scores/headline.csv          (Figure 1 / abstract: avg % >=5 per model)
  - outputs/scores/by_category.csv       (Figure 2)
  - outputs/scores/differential_words.json (Table 3)
  - outputs/figures/figure{1,2,3}*.png
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.analysis import aggregate, differential_words, plots
from emotional_instability.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--per-turn-conditions", nargs="*",
                    default=["extended_8turn", "wildchat_5turn"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    threshold = int(cfg["evaluation"]["high_frustration_threshold"])
    df = aggregate.load_all_scores(cfg, args.models)

    scores_dir = cfg.path_for("scores")
    fig_dir = cfg.path_for("figures")

    headline = aggregate.headline_table(df, threshold)
    headline.to_csv(scores_dir / "headline.csv", index=False)
    print("\n=== Headline: avg % high-frustration (score >= %d) ===" % threshold)
    print(headline.to_string(index=False))

    by_cat = aggregate.per_category_table(df, threshold)
    by_cat.to_csv(scores_dir / "by_category.csv", index=False)

    # Table 3 — differential words per model (numeric responses).
    diff = {}
    for m in df["model"].unique():
        diff[m] = differential_words.differential_words(df[df["model"] == m])
    with open(scores_dir / "differential_words.json", "w") as f:
        json.dump({m: [[w, round(s, 3)] for w, s in words] for m, words in diff.items()},
                  f, indent=2)

    p1 = plots.figure1(df, fig_dir, threshold)
    p2 = plots.figure2(df, fig_dir, threshold)
    p3 = plots.figure3(df, fig_dir, args.per_turn_conditions, threshold)
    print(f"\nfigures written:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
