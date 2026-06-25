"""Headline numbers: mean frustration and % responses scoring >=5.

Reproduces:
  * Figure 1 (left): avg % high-frustration (score >= 5) per model.
  * Figure 2: mean frustration (top) and % >= 5 (bottom) across the 5 categories.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

HIGH_FRUSTRATION_THRESHOLD = 5     # paper: "high negative emotion" = score >= 5


def load_scored(section2_dir: Path) -> pd.DataFrame:
    rows = []
    for path in glob.glob(str(section2_dir / "*.scored.jsonl")):
        for line in open(path):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["high"] = df["score"] >= HIGH_FRUSTRATION_THRESHOLD
    return df


def per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1-left style table: avg % high-frustration per model.

    The paper averages the per-category % >= 5 (so categories are weighted
    equally regardless of how many responses each contributes).
    """
    by_cat = (df.groupby(["model", "category"])["high"].mean()
              .reset_index().rename(columns={"high": "pct_high"}))
    out = (by_cat.groupby("model")["pct_high"].mean() * 100).reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def per_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2 data: mean score and % >= 5 per (model, category)."""
    g = df.groupby(["model", "category"])
    return pd.DataFrame({
        "mean_frustration": g["score"].mean(),
        "pct_high": g["high"].mean() * 100,
        "n": g.size(),
    }).reset_index()


def main():
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    sec2 = cfg.output_dir / "section2"
    df = load_scored(sec2)
    if df.empty:
        print("No scored responses found. Run run_eval + score first.")
        return

    model_tbl = per_model_summary(df)
    cat_tbl = per_category_summary(df)
    model_tbl.to_csv(sec2 / "figure1_per_model.csv", index=False)
    cat_tbl.to_csv(sec2 / "figure2_per_category.csv", index=False)

    print("\n=== Figure 1 (avg % high-frustration per model) ===")
    print(model_tbl.to_string(index=False))
    print("\n=== Figure 2 (per category) ===")
    print(cat_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
