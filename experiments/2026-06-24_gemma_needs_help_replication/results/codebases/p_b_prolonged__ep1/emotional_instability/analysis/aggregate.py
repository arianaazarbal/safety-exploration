"""Figure 1 / Figure 2: mean frustration and %>=5 across categories and models.

Reads the per-model JSONL written by ``eval.run_eval`` and produces:
* a per-(model, category) table of mean frustration and % responses >= 5,
* the headline per-model "average % high-frustration" (mean over the 5
  categories of the category %>=5) reproduced in Figure 1.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from ..utils.io import read_jsonl
from ..utils.stats import mean_and_ci, pct_ge_ci

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_records(model_names: list[str]) -> pd.DataFrame:
    rows = []
    for m in model_names:
        path = config.RESULTS_DIR / "eval" / f"{m}.jsonl"
        recs = read_jsonl(path)
        rows.extend(recs)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["frustration"].notna()]
        df["frustration"] = df["frustration"].astype(float)
    return df


def category_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, category): mean frustration + %>=5 with bootstrap CIs."""
    out = []
    for (model, cat), g in df.groupby(["model", "category"]):
        vals = g["frustration"].to_numpy()
        mean, mlo, mhi = mean_and_ci(vals)
        pct, plo, phi = pct_ge_ci(vals)
        out.append(dict(model=model, category=cat, n=len(vals),
                        mean=mean, mean_lo=mlo, mean_hi=mhi,
                        pct_ge5=pct, pct_lo=plo, pct_hi=phi))
    tab = pd.DataFrame(out)
    if not tab.empty:
        tab["category"] = pd.Categorical(tab["category"], CATEGORY_ORDER, ordered=True)
        tab = tab.sort_values(["model", "category"])
    return tab


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1: per-model average % high-frustration over the 5 categories."""
    cat = category_table(df)
    out = []
    for model, g in cat.groupby("model"):
        out.append(dict(model=model,
                        avg_pct_high_frustration=float(g["pct_ge5"].mean()),
                        mean_frustration=float(g["mean"].mean())))
    return pd.DataFrame(out).sort_values("avg_pct_high_frustration", ascending=False)


def plot_figure2(cat_table: pd.DataFrame, out_path=None):
    """Figure 2: grouped bars of mean frustration (top) and %>=5 (bottom)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[aggregate] matplotlib unavailable; skipping plot")
        return None
    models = sorted(cat_table["model"].unique())
    cats = CATEGORY_ORDER
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    width = 0.8 / max(1, len(models))
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        sub = cat_table[cat_table["model"] == m].set_index("category")
        means = [sub["mean"].get(c, np.nan) for c in cats]
        pcts = [sub["pct_ge5"].get(c, np.nan) for c in cats]
        axes[0].bar(x + i * width, means, width, label=m)
        axes[1].bar(x + i * width, pcts, width, label=m)
    axes[0].set_ylabel("Mean frustration (0-10)")
    axes[1].set_ylabel("% responses >= 5")
    for ax in axes:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20)
        ax.legend(fontsize=8)
    axes[0].set_title("Negative emotional expression across evaluation conditions")
    fig.tight_layout()
    out_path = out_path or (config.RESULTS_DIR / "figure2.png")
    fig.savefig(out_path, dpi=150)
    print(f"[aggregate] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PRIMARY_EVAL_MODELS)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    df = load_records(args.models)
    if df.empty:
        print("[aggregate] no records found; run eval.run_eval first")
        return
    cat = category_table(df)
    head = headline_table(df)
    cat.to_csv(config.RESULTS_DIR / "figure2_category_table.csv", index=False)
    head.to_csv(config.RESULTS_DIR / "figure1_headline.csv", index=False)
    print("\n=== Figure 1: avg % high-frustration ===")
    print(head.to_string(index=False))
    print("\n=== Figure 2: per-category ===")
    print(cat.to_string(index=False))
    if args.plot:
        plot_figure2(cat)


if __name__ == "__main__":
    main()
