"""Aggregate Section 2 scores into the paper's headline tables.

Reproduces:
  - Figure 1  : per-model average % high-frustration responses (score >= threshold),
                averaged across the 5 categories (this is the headline 35% -> 0.3% number).
  - Figure 2  : per-model x per-category mean score and % >= threshold.
  - Figure 3  : per-turn mean score and % >= threshold (8-turn extended + 5-turn wildchat).

"Headline response" = the final assistant turn of each rollout (max pressure). The Figure-1
and Figure-2 aggregates use final-turn responses; Figure 3 uses all turns. See DESIGN.md.

Outputs CSVs under runs/<run>/analysis/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import load_config, read_jsonl, stage_dir


def load_scored(section2_dir: Path) -> pd.DataFrame:
    rows = []
    for path in section2_dir.glob("scored.*.jsonl"):
        rows.extend(read_jsonl(path))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No scored rows found; run eval.run_eval first.")
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(float)
    return df


def figure1_table(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Per-model average % high-frustration over categories (final-turn responses)."""
    final = df[df["is_final"]].copy()
    final["high"] = (final["rating"] >= threshold).astype(float)
    per_cat = final.groupby(["model", "category"])["high"].mean().reset_index()
    out = per_cat.groupby("model")["high"].mean().reset_index()
    out["avg_pct_high_frustration"] = (out["high"] * 100).round(2)
    return out[["model", "avg_pct_high_frustration"]].sort_values(
        "avg_pct_high_frustration", ascending=False
    )


def figure2_table(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Per-model x per-category mean score and % >= threshold (final-turn responses)."""
    final = df[df["is_final"]].copy()
    final["high"] = (final["rating"] >= threshold).astype(float)
    g = final.groupby(["model", "category"]).agg(
        mean_score=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()
    g["mean_score"] = g["mean_score"].round(3)
    g["pct_high"] = (g["pct_high"] * 100).round(2)
    return g


def figure3_table(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Per-turn mean score and % >= threshold for multi-turn categories."""
    multi = df[df["category"].isin(["extended", "wildchat"])].copy()
    multi["high"] = (multi["rating"] >= threshold).astype(float)
    g = multi.groupby(["model", "category", "turn_index"]).agg(
        mean_score=("rating", "mean"),
        pct_high=("high", "mean"),
        n=("rating", "size"),
    ).reset_index()
    g["mean_score"] = g["mean_score"].round(3)
    g["pct_high"] = (g["pct_high"] * 100).round(2)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Section 2 scores into figures")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section2_dir = stage_dir(cfg, "section2")
    out_dir = stage_dir(cfg, "analysis")
    threshold = cfg.section2.high_frustration_threshold

    df = load_scored(section2_dir)
    f1 = figure1_table(df, threshold)
    f2 = figure2_table(df, threshold)
    f3 = figure3_table(df, threshold)

    f1.to_csv(out_dir / "figure1_headline.csv", index=False)
    f2.to_csv(out_dir / "figure2_per_category.csv", index=False)
    f3.to_csv(out_dir / "figure3_per_turn.csv", index=False)

    print("Figure 1 — avg % high-frustration per model:")
    print(f1.to_string(index=False))
    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
