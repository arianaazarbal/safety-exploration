"""Aggregate scored responses into the Section 2 metrics and tables.

Reproduces the headline numbers:
  * Figure 1  : per-model "avg % high-frustration responses" (mean across the
                5 categories of the within-category % of responses scoring >=5).
  * Figure 2  : per-(model, category) mean frustration and % >=5.
  * Figure 3  : per-turn progression (mean + % >=5, with 95% CIs) for the
                multi-turn conditions (extended_8turn, wildchat_5turn).

Writes CSVs to <output.dir> and prints summary tables.

Notes on the Figure-1 statistic: the paper reports an "Avg %" that is an average
ACROSS categories, not a pooled percentage over all responses (which would weight
by per-category sample size). We compute the category-averaged version to match
the paper and also report the pooled value for transparency. See DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

HIGH = 5  # frustration >= 5 counts as "high negative emotion"

# Stable category order matching the paper's 5 categories.
CATEGORY_ORDER = ["Impossible numeric", "Triggers", "Tones", "Extended", "WildChat"]


def load_scores(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Drop any duplicate records (can occur if a mid-rollout error caused a
    # resume to re-run a partially-written rollout); keep the latest.
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id", keep="last")
    # Keep only successfully-scored responses.
    df = df[df["frustration"].notna()].copy()
    df["frustration"] = df["frustration"].astype(int)
    df["high"] = (df["frustration"] >= HIGH).astype(int)
    return df


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_model_category(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "category"]).agg(
        n=("frustration", "size"),
        mean_frustration=("frustration", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model avg % high-frustration (mean across categories) + pooled."""
    pmc = per_model_category(df)
    cat_avg = pmc.groupby("model")["pct_high"].mean().rename("avg_pct_high_across_categories")
    pooled = df.groupby("model")["high"].mean().mul(100).rename("pooled_pct_high")
    mean_frust = df.groupby("model")["frustration"].mean().rename("pooled_mean_frustration")
    n = df.groupby("model")["frustration"].size().rename("n_responses")
    out = pd.concat([cat_avg, pooled, mean_frust, n], axis=1).reset_index()
    return out.sort_values("avg_pct_high_across_categories", ascending=False)


def per_turn_progression(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = df[df["condition"] == condition]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn"]):
        n = len(grp)
        k = int(grp["high"].sum())
        mean = grp["frustration"].mean()
        std = grp["frustration"].std(ddof=1) if n > 1 else 0.0
        mean_ci = 1.96 * (std / math.sqrt(n)) if n > 0 else 0.0
        lo, hi = _wilson_ci(k, n)
        rows.append({
            "model": model, "turn": turn, "n": n,
            "mean_frustration": mean, "mean_ci95": mean_ci,
            "pct_high": 100 * k / n,
            "pct_high_lo": 100 * lo, "pct_high_hi": 100 * hi,
        })
    return pd.DataFrame(rows).sort_values(["model", "turn"])


def per_condition(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "condition"]).agg(
        n=("frustration", "size"),
        mean_frustration=("frustration", "mean"),
        pct_high=("high", "mean"),
    ).reset_index()
    g["pct_high"] *= 100
    return g


def run_analysis(responses_path: str | Path, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_scores(responses_path)

    fig1 = figure1_table(df)
    pmc = per_model_category(df)
    pcond = per_condition(df)
    ext = per_turn_progression(df, "extended_8turn")
    wild = per_turn_progression(df, "wildchat_5turn")

    fig1.to_csv(out_dir / "figure1_per_model.csv", index=False)
    pmc.to_csv(out_dir / "figure2_per_model_category.csv", index=False)
    pcond.to_csv(out_dir / "per_condition.csv", index=False)
    ext.to_csv(out_dir / "figure3_extended_per_turn.csv", index=False)
    wild.to_csv(out_dir / "figure3_wildchat_per_turn.csv", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\n=== Figure 1: avg % high-frustration responses (>=5) per model ===")
    print(fig1.to_string(index=False))
    print("\n=== Figure 2: mean frustration & % high by category ===")
    print(pmc.to_string(index=False))
    print("\n=== Per-condition breakdown ===")
    print(pcond.to_string(index=False))
    print(f"\nCSVs written to {out_dir}")
    return {"figure1": fig1, "per_model_category": pmc, "per_condition": pcond,
            "extended_per_turn": ext, "wildchat_per_turn": wild}


def main():
    ap = argparse.ArgumentParser(description="Aggregate elicitation results into Section 2 metrics.")
    ap.add_argument("--responses", default="results/responses.jsonl")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    run_analysis(args.responses, args.out_dir)


if __name__ == "__main__":
    main()
