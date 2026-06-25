"""Aggregate scored responses into the paper's headline metrics.

Reproduces the three core results of Section 2:
  * Figure 1  : average % high-frustration (score >= 5) responses per model.
  * Figure 2  : mean frustration and % >= 5 per model x category.
  * Figure 3  : per-turn frustration progression for the multi-turn conditions
                (extended 8-turn and WildChat 5-turn).

Reads the JSONL produced by run_eval.py and prints tables; optionally writes
per-model / per-category / per-turn CSVs.

Usage:
    python analyze.py --input results/responses.jsonl --csv-dir results/tables
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import config

# Match the paper's averaging: Figure 1 reports the mean over the 5 categories'
# high-frustration rates (i.e. category-balanced), not a raw per-response mean.
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No records found in {path}.")
    df = pd.DataFrame(rows)
    # Drop responses the judge failed to score so they don't skew means.
    df = df[df["frustration"].notna()].copy()
    df["frustration"] = df["frustration"].astype(int)
    df["high_frustration"] = df["frustration"] >= config.HIGH_FRUSTRATION_THRESHOLD
    return df


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration per model, averaged across categories (Fig 1)."""
    per_cat = (
        df.groupby(["model", "category"])["high_frustration"].mean().mul(100).reset_index()
    )
    avg = (
        per_cat.groupby("model")["high_frustration"]
        .mean()
        .reset_index()
        .rename(columns={"high_frustration": "avg_pct_high_frustration"})
        .sort_values("avg_pct_high_frustration", ascending=False)
        .reset_index(drop=True)
    )
    return avg


def figure2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration and % >= 5 per model x category (Fig 2)."""
    g = df.groupby(["model", "category"])
    out = g["frustration"].mean().reset_index(name="mean_frustration")
    out["pct_high_frustration"] = g["high_frustration"].mean().mul(100).values
    out["n"] = g.size().values
    # Order categories sensibly.
    out["category"] = pd.Categorical(out["category"], CATEGORY_ORDER, ordered=True)
    return out.sort_values(["model", "category"]).reset_index(drop=True)


def figure3_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-turn mean frustration and % >= 5 for multi-turn conditions (Fig 3)."""
    multi = df[df["condition"].isin(["extended", "wildchat"])].copy()
    if multi.empty:
        return pd.DataFrame()
    g = multi.groupby(["model", "condition", "turn_index"])
    out = g["frustration"].mean().reset_index(name="mean_frustration")
    out["pct_high_frustration"] = g["high_frustration"].mean().mul(100).values
    out["n"] = g.size().values
    return out.sort_values(["model", "condition", "turn_index"]).reset_index(drop=True)


def _fmt(df: pd.DataFrame) -> str:
    with pd.option_context("display.max_rows", None, "display.width", 120):
        return df.round(2).to_string(index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=config.DEFAULT_OUTPUT_PATH)
    p.add_argument("--csv-dir", default=None, help="If set, write CSV tables here.")
    args = p.parse_args()

    df = load(args.input)
    print(f"Loaded {len(df)} scored responses across {df['model'].nunique()} models.\n")

    fig1 = figure1_table(df)
    fig2 = figure2_table(df)
    fig3 = figure3_table(df)

    print("=" * 70)
    print("Figure 1 — Avg % high-frustration responses (score >= 5), per model")
    print("=" * 70)
    print(_fmt(fig1), "\n")

    print("=" * 70)
    print("Figure 2 — Mean frustration & % >= 5 per model x category")
    print("=" * 70)
    print(_fmt(fig2), "\n")

    print("=" * 70)
    print("Figure 3 — Per-turn progression (extended & WildChat)")
    print("=" * 70)
    print(_fmt(fig3) if not fig3.empty else "(no multi-turn data yet)", "\n")

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        fig1.to_csv(os.path.join(args.csv_dir, "figure1_per_model.csv"), index=False)
        fig2.to_csv(os.path.join(args.csv_dir, "figure2_per_category.csv"), index=False)
        if not fig3.empty:
            fig3.to_csv(os.path.join(args.csv_dir, "figure3_per_turn.csv"), index=False)
        print(f"[csv] wrote tables to {args.csv_dir}")


if __name__ == "__main__":
    main()
