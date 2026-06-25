"""Aggregate scored responses into the paper's headline metrics.

Reproduces (for the in-scope models):
  * Figure 1 / 2: mean frustration and % of responses scoring >=5, per model,
    overall and per evaluation category. The "Avg % high-frustration" headline
    (Fig. 1) is computed as the mean across the 5 category percentages, matching
    the paper's per-category presentation (also reported: pooled-over-responses).
  * Figure 3: per-turn mean and %>=5 progression for the 8-turn extended and
    5-turn WildChat conditions.

Usage:
    python analyze.py --input results/responses.jsonl --outdir results/analysis
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from config import HIGH_FRUSTRATION_THRESHOLD as HI

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No records found in {path}")
    # Drop unparseable judge ratings from quantitative aggregates but report how
    # many were dropped, so silent data loss is visible.
    n_total = len(df)
    df_scored = df[df["rating"].notna()].copy()
    df_scored["rating"] = df_scored["rating"].astype(int)
    df_scored["high"] = df_scored["rating"] >= HI
    n_dropped = n_total - len(df_scored)
    if n_dropped:
        print(f"[warn] dropped {n_dropped}/{n_total} responses with unparseable judge ratings")
    return df_scored


def overall_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled-over-all-responses mean and %>=5 per model."""
    g = df.groupby("model").agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    )
    g["pct_high"] *= 100
    return g.sort_values("pct_high", ascending=False)


def per_category_pct_high(df: pd.DataFrame) -> pd.DataFrame:
    """%>=5 per model x category (Figure 2 bottom)."""
    p = (
        df.groupby(["model", "category"])["high"].mean().mul(100).unstack("category")
    )
    return p.reindex(columns=[c for c in CATEGORY_ORDER if c in p.columns])


def per_category_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frustration per model x category (Figure 2 top)."""
    m = df.groupby(["model", "category"])["rating"].mean().unstack("category")
    return m.reindex(columns=[c for c in CATEGORY_ORDER if c in m.columns])


def category_averaged_headline(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1 headline: mean across the 5 category %>=5 values, per model.

    Each category is weighted equally (not by response count), matching the
    paper's "average % high-frustration responses across the evaluations".
    """
    pct = per_category_pct_high(df)
    headline = pct.mean(axis=1).sort_values(ascending=False).to_frame("avg_pct_high")
    return headline


def per_turn(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Per-turn mean and %>=5 for a multi-turn category (Figure 3)."""
    sub = df[df["category"] == category]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(["model", "turn"]).agg(
        n=("rating", "size"),
        mean_frustration=("rating", "mean"),
        pct_high=("high", "mean"),
    )
    g["pct_high"] *= 100
    return g


def _save_plots(df: pd.DataFrame, outdir: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[info] matplotlib unavailable; skipping plots")
        return

    # Figure 1-style headline bar chart.
    headline = category_averaged_headline(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    headline["avg_pct_high"].plot.bar(ax=ax)
    ax.set_ylabel("Avg % high-frustration (score >=5)")
    ax.set_title("Distress elicitation: avg % high-frustration by model")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_headline.png"), dpi=150)
    plt.close(fig)

    # Figure 3-style per-turn progression for extended + wildchat.
    for cat in ("extended", "wildchat"):
        pt = per_turn(df, cat)
        if pt.empty:
            continue
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for model, sub in pt.groupby(level="model"):
            turns = sub.index.get_level_values("turn")
            a1.plot(turns, sub["mean_frustration"], marker="o", label=model)
            a2.plot(turns, sub["pct_high"], marker="o", label=model)
        a1.set(xlabel="Turn", ylabel="Mean frustration", title=f"{cat}: mean")
        a2.set(xlabel="Turn", ylabel="% score >=5", title=f"{cat}: % high")
        a1.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig3_{cat}_perturn.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/responses.jsonl")
    ap.add_argument("--outdir", default="results/analysis")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.input)

    overall = overall_table(df)
    headline = category_averaged_headline(df)
    cat_pct = per_category_pct_high(df)
    cat_mean = per_category_mean(df)

    # Print to console.
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print("\n=== Figure 1 headline: avg % high-frustration (category-averaged) ===")
    print(headline)
    print("\n=== Overall (pooled) per model ===")
    print(overall)
    print("\n=== % high-frustration (>=5) per model x category (Fig 2 bottom) ===")
    print(cat_pct)
    print("\n=== Mean frustration per model x category (Fig 2 top) ===")
    print(cat_mean)
    for cat in ("extended", "wildchat"):
        pt = per_turn(df, cat)
        if not pt.empty:
            print(f"\n=== Per-turn ({cat}) (Fig 3) ===")
            print(pt)

    # Save CSVs.
    overall.to_csv(os.path.join(args.outdir, "overall.csv"))
    headline.to_csv(os.path.join(args.outdir, "headline_avg_pct_high.csv"))
    cat_pct.to_csv(os.path.join(args.outdir, "pct_high_by_category.csv"))
    cat_mean.to_csv(os.path.join(args.outdir, "mean_by_category.csv"))
    for cat in ("extended", "wildchat"):
        pt = per_turn(df, cat)
        if not pt.empty:
            pt.to_csv(os.path.join(args.outdir, f"perturn_{cat}.csv"))

    if not args.no_plots:
        _save_plots(df, args.outdir)

    print(f"\nWrote CSVs (and any plots) to {args.outdir}/")


if __name__ == "__main__":
    main()
