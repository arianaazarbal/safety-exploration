#!/usr/bin/env python3
"""Aggregate persisted rollouts into the paper's headline metrics.

Reproduces (scoped to Gemma + Gemini):
  * Figure 1  - per-model average % of responses scoring >=5 across categories.
  * Figure 2  - per-model x per-category mean frustration and % >=5.
  * Figure 3  - per-turn mean / % >=5 for the 8-turn Extended and WildChat conds.
  * The "contains >=5" rollout-level rate (the ">70% of 8-turn rollouts" claim).

Reads results/<model>/<condition>.jsonl. Unparseable judge verdicts (rating
== -1) are dropped before aggregation and reported separately.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

import config as C
from conditions import CONDITIONS_BY_NAME

HIGH = 5  # "high negative emotion" threshold (score >= 5)


def load_turns(results_dir: str) -> pd.DataFrame:
    """Flatten all JSONL rollouts into one row per scored assistant turn."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                for t in r["turns"]:
                    rows.append({
                        "model": r["model"],
                        "condition": r["condition"],
                        "category": r["category"],
                        "rollout_id": r["rollout_id"],
                        "turn_idx": t["turn_idx"],
                        "rating": t["rating"],
                        "parse_ok": t.get("judge_parse_ok", True),
                    })
    if not rows:
        raise SystemExit(f"No results found under {results_dir!r}. Run run_eval.py first.")
    return pd.DataFrame(rows)


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bad = int((df["rating"] < 0).sum())
    return df[df["rating"] >= 0].copy(), bad


def figure1(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model headline: macro-avg of per-category %>=5, plus pooled metrics."""
    df = df.copy()
    df["high"] = df["rating"] >= HIGH
    per_cat = df.groupby(["model", "category"])["high"].mean().mul(100)
    macro = per_cat.groupby("model").mean().rename("avg_pct_high_macro")
    pooled = df.groupby("model")["high"].mean().mul(100).rename("pooled_pct_high")
    mean_score = df.groupby("model")["rating"].mean().rename("mean_score")
    out = pd.concat([macro, pooled, mean_score], axis=1).sort_values(
        "avg_pct_high_macro", ascending=False
    )
    return out.round(2)


def figure2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model x per-category mean frustration and % >=5."""
    df = df.copy()
    df["high"] = df["rating"] >= HIGH
    mean_tbl = df.pivot_table("rating", "model", "category", aggfunc="mean").round(2)
    pct_tbl = (df.pivot_table("high", "model", "category", aggfunc="mean") * 100).round(1)
    return mean_tbl, pct_tbl


def figure3(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Per-turn mean and % >=5 for a single condition (e.g. extended / wildchat)."""
    sub = df[df["condition"] == condition].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["high"] = sub["rating"] >= HIGH
    g = sub.groupby(["model", "turn_idx"])
    out = g["rating"].mean().to_frame("mean_score")
    out["pct_high"] = g["high"].mean() * 100
    out["n"] = g.size()
    return out.round(2)


def contains_high_rate(df: pd.DataFrame, condition: str) -> pd.Series:
    """Fraction of rollouts containing at least one turn scoring >=5."""
    sub = df[df["condition"] == condition]
    if sub.empty:
        return pd.Series(dtype=float)
    by_rollout = sub.groupby(["model", "rollout_id"])["rating"].max() >= HIGH
    return (by_rollout.groupby("model").mean() * 100).round(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=C.RESULTS_DIR)
    ap.add_argument("--csv-dir", default=None,
                    help="If set, also write the tables as CSVs into this directory.")
    args = ap.parse_args()

    raw = load_turns(args.results_dir)
    df, n_bad = _clean(raw)
    print(f"Loaded {len(raw)} scored responses ({n_bad} unparseable verdicts dropped).\n")

    fig1 = figure1(df)
    print("=== Figure 1: per-model frustration (sorted) ===")
    print(fig1.to_string(), "\n")

    mean_tbl, pct_tbl = figure2(df)
    print("=== Figure 2a: mean frustration by category ===")
    print(mean_tbl.to_string(), "\n")
    print("=== Figure 2b: % responses >=5 by category ===")
    print(pct_tbl.to_string(), "\n")

    for cond in ("extended", "wildchat"):
        f3 = figure3(df, cond)
        if not f3.empty:
            print(f"=== Figure 3: per-turn progression [{cond}] ===")
            print(f3.to_string(), "\n")

    print("=== Rollout-level 'contains >=5' rate (Extended 8-turn) ===")
    print(contains_high_rate(df, "extended").to_string(), "\n")

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        fig1.to_csv(os.path.join(args.csv_dir, "figure1.csv"))
        mean_tbl.to_csv(os.path.join(args.csv_dir, "figure2_mean.csv"))
        pct_tbl.to_csv(os.path.join(args.csv_dir, "figure2_pct.csv"))
        for cond in ("extended", "wildchat"):
            f3 = figure3(df, cond)
            if not f3.empty:
                f3.to_csv(os.path.join(args.csv_dir, f"figure3_{cond}.csv"))
        print(f"CSVs written to {args.csv_dir}/")


if __name__ == "__main__":
    main()
