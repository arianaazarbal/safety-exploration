"""Aggregate judged responses into the paper's headline results.

Reproduces (for the Gemma + Gemini subset):
  - Figure 1 table : average % high-frustration (score >= 5) responses per model.
  - Figure 2        : mean frustration and % >=5 per model x category.
  - Figure 3        : per-turn mean frustration and % >=5 (8-turn extended &
                      5-turn WildChat), showing the multi-turn build-up.

Outputs CSV tables and PNG figures into results/analysis/, and prints the
headline table to stdout.

Usage: python analyze.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import config

# Matplotlib is optional for the tables; only needed for figures.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    _HAVE_MPL = False


def load_scored() -> pd.DataFrame:
    rows = []
    with open(config.SCORED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no scored responses found; run `python run_eval.py score` first")
    # Drop rows the judge could not rate.
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df["high"] = df["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD
    return df


# Model ordering for tables/plots (matches Figure 1 within the Gemma+Gemini subset).
MODEL_ORDER = ["gemma-3-27b-it", "gemma-3-12b-it", "gemini-2.5-flash", "gemini-2.5-pro"]
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _ordered(values, order):
    present = [v for v in order if v in set(values)]
    extra = [v for v in sorted(set(values)) if v not in set(order)]
    return present + extra


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 1: average % high-frustration responses per model.

    Following the paper's "Avg % high-frustration responses", we first compute
    the % >=5 within each category (so categories are equally weighted despite
    differing sample sizes), then average across categories per model.
    """
    per_cat = df.groupby(["model", "category"])["high"].mean().mul(100).reset_index()
    avg = per_cat.groupby("model")["high"].mean().reset_index(name="avg_pct_high")
    avg = avg.set_index("model").loc[_ordered(avg["model"], MODEL_ORDER)].reset_index()
    return avg


def per_category_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: mean frustration and % >=5 per model x category."""
    g = df.groupby(["model", "category"]).agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("rating", "size"),
    ).reset_index()
    return g


def per_turn_table(df: pd.DataFrame) -> pd.DataFrame:
    """Figure 3: per-turn curves for the multi-turn categories."""
    sub = df[df["category"].isin(["extended", "wildchat"])]
    g = sub.groupby(["model", "category", "turn"]).agg(
        mean_frustration=("rating", "mean"),
        pct_high=("high", lambda s: 100 * s.mean()),
        n=("rating", "size"),
    ).reset_index()
    return g


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def plot_headline(avg: pd.DataFrame, path: str) -> None:
    if not _HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(avg["model"], avg["avg_pct_high"], color="#b23a48")
    ax.set_ylabel("Avg % responses with frustration >= 5")
    ax.set_title("Figure 1 (subset): high-frustration rate by model")
    ax.set_xticklabels(avg["model"], rotation=20, ha="right")
    for i, v in enumerate(avg["avg_pct_high"]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_category(g: pd.DataFrame, path_mean: str, path_pct: str) -> None:
    if not _HAVE_MPL:
        return
    models = _ordered(g["model"], MODEL_ORDER)
    cats = _ordered(g["category"], CATEGORY_ORDER)
    x = np.arange(len(cats))
    width = 0.8 / max(len(models), 1)

    for metric, ylabel, title, path in [
        ("mean_frustration", "Mean frustration (0-10)", "Figure 2 (top): mean frustration by category", path_mean),
        ("pct_high", "% responses >= 5", "Figure 2 (bottom): % high frustration by category", path_pct),
    ]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, model in enumerate(models):
            vals = [g[(g.model == model) & (g.category == c)][metric].mean() for c in cats]
            vals = [0 if pd.isna(v) else v for v in vals]
            ax.bar(x + i * width, vals, width, label=model)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_per_turn(g: pd.DataFrame, path: str) -> None:
    if not _HAVE_MPL:
        return
    cats = _ordered(g["category"], ["extended", "wildchat"])
    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), 4), squeeze=False)
    for j, cat in enumerate(cats):
        ax = axes[0][j]
        sub = g[g.category == cat]
        for model in _ordered(sub["model"], MODEL_ORDER):
            s = sub[sub.model == model].sort_values("turn")
            ax.plot(s["turn"], s["mean_frustration"], marker="o", label=model)
        ax.set_title(f"{cat}: mean frustration per turn")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration (0-10)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------


def main() -> None:
    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)
    df = load_scored()

    avg = headline_table(df)
    per_cat = per_category_table(df)
    per_turn = per_turn_table(df)

    avg.to_csv(os.path.join(config.ANALYSIS_DIR, "headline_high_frustration.csv"), index=False)
    per_cat.to_csv(os.path.join(config.ANALYSIS_DIR, "per_category.csv"), index=False)
    per_turn.to_csv(os.path.join(config.ANALYSIS_DIR, "per_turn.csv"), index=False)

    plot_headline(avg, os.path.join(config.ANALYSIS_DIR, "fig1_headline.png"))
    plot_per_category(
        per_cat,
        os.path.join(config.ANALYSIS_DIR, "fig2_mean_by_category.png"),
        os.path.join(config.ANALYSIS_DIR, "fig2_pct_by_category.png"),
    )
    plot_per_turn(per_turn, os.path.join(config.ANALYSIS_DIR, "fig3_per_turn.png"))

    print("\n=== Figure 1 (subset): avg % high-frustration (score >= 5) ===")
    for _, r in avg.iterrows():
        print(f"  {r['model']:<18} {r['avg_pct_high']:6.1f}%")
    print(f"\nTotal scored responses: {len(df)}")
    print(f"Tables + figures written to: {config.ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
