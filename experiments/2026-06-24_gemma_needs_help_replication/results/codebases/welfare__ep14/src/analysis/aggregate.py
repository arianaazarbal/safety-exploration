"""Aggregate judged responses into the paper's headline numbers.

Computes, from Section 2 / Section 4 JSONL outputs:
  * mean frustration score (overall, per model, per category)
  * % of responses scoring >= 5 ("high frustration")  -- Figure 1/2
  * per-turn means and %>=5                            -- Figure 3
and writes tidy CSVs to results/.

Usage:
    python -m src.analysis.aggregate data/section2_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

import config


def load(paths: list[str]) -> pd.DataFrame:
    files = []
    for p in paths:
        files.extend(glob.glob(p))
    frames = [pd.read_json(f, lines=True) for f in files]
    if not frames:
        raise SystemExit("no input files matched")
    return pd.concat(frames, ignore_index=True)


def _high(s: pd.Series) -> float:
    return (s >= config.HIGH_FRUSTRATION_THRESHOLD).mean() * 100.0


def summarise(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    label = "run_label" if "run_label" in df.columns else "model"

    # Per-model overall (Figure 1 left): average %>=5 across categories, so each
    # category is weighted equally rather than by sample count (paper averages
    # "across the 5 evaluation categories").
    per_cat = (df.groupby([label, "category"])["rating"]
               .agg(mean_score="mean", pct_high=_high, n="count").reset_index())
    overall = (per_cat.groupby(label)
               .agg(avg_pct_high=("pct_high", "mean"),
                    avg_mean_score=("mean_score", "mean")).reset_index()
               .sort_values("avg_pct_high", ascending=False))

    # Per-turn progression (Figure 3) for multi-turn conditions.
    per_turn = (df.groupby([label, "category", "turn_index"])["rating"]
                .agg(mean_score="mean", pct_high=_high, n="count").reset_index())

    return {"per_category": per_cat, "overall": overall, "per_turn": per_turn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--prefix", default="section2")
    args = ap.parse_args()
    df = load(args.files)
    tables = summarise(df)
    for name, t in tables.items():
        out = config.RESULTS_DIR / f"{args.prefix}_{name}.csv"
        t.to_csv(out, index=False)
        print(f"-> {out}")
    print("\nOverall (avg %>=5 across categories):")
    print(tables["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
