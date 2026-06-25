#!/usr/bin/env python
"""Aggregate persisted results into the paper's tables/figures.

Reads results/*.csv and emits figures under figures/ plus a printed headline
table (the Figure-1 metric). Safe to run after any subset of experiments; it
plots whatever is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from distress import config  # noqa: E402
from distress.analysis import figures  # noqa: E402
from distress.eval.metrics import (  # noqa: E402
    headline_pct_high,
    per_category_metrics,
    per_turn_metrics,
)


def _load_section2(tagged: bool = False) -> pd.DataFrame:
    pattern = "section2_*_scored.csv"
    frames = []
    for p in sorted(config.RESULTS_DIR.glob(pattern)):
        df = pd.read_csv(p)
        # Encode the variant tag (e.g. -dpo) into the model name for Figure 5.
        stem = p.stem.replace("section2_", "").replace("_scored", "")
        df["model"] = stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    df = _load_section2()
    if df.empty:
        print("No Section 2 results found. Run scripts/run_section2.py first.")
    else:
        headline = headline_pct_high(df)
        headline.to_csv(config.RESULTS_DIR / "headline_pct_high.csv", index=False)
        print("\n=== Figure 1: avg % high-frustration (score >= 5) ===")
        print(headline.to_string(index=False))

        cat = per_category_metrics(df)
        cat.to_csv(config.RESULTS_DIR / "per_category_metrics.csv", index=False)

        # Plot only the canonical Section-2 model set for Figures 1-3.
        canonical = df[df["model"].isin([s.key for s in config.SECTION2_MODELS])]
        if not canonical.empty:
            figures.fig1_headline(headline_pct_high(canonical))
            figures.fig2_per_category(per_category_metrics(canonical))
            turns = per_turn_metrics(canonical, categories=["extended", "wildchat"])
            figures.fig3_per_turn(turns)

        # Figure 5: finetune variants (any model name containing gemma...-<tag>).
        variants = df[df["model"].str.startswith("gemma-3-27b-it")]
        if variants["model"].nunique() > 1:
            figures.fig5_finetune(per_category_metrics(variants))

    # Section 3.
    s3 = config.RESULTS_DIR / "section3_metrics.csv"
    if s3.exists():
        figures.fig4_prefill(pd.read_csv(s3))

    # Petri.
    petri = config.RESULTS_DIR / "petri_metrics.csv"
    if petri.exists():
        figures.fig6_petri(pd.read_csv(petri))

    # Capabilities.
    cap = config.RESULTS_DIR / "capabilities.csv"
    if cap.exists():
        figures.fig7_capabilities(pd.read_csv(cap))

    print(f"\nFigures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
