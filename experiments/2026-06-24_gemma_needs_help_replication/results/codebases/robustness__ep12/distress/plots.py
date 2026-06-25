"""Figure generation (matplotlib). Mirrors the paper's key figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import analysis  # noqa: E402


def fig1_ranking(df, out_path):
    """Bar chart: average % >=5 per model (Figure 1 / headline table)."""
    summ = analysis.per_model_summary(df).sort_values("cw_pct_high")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(summ) + 1))
    ax.barh(summ["model"], summ["cw_pct_high"])
    ax.set_xlabel("Avg % responses scoring >=5 (high frustration)")
    ax.set_title("Emotional instability by model")
    for y, v in enumerate(summ["cw_pct_high"]):
        ax.text(v, y, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_categories(df, out_path):
    """Grouped bars: % >=5 per (model, category) (Figure 2)."""
    g = analysis.per_category_summary(df)
    models = sorted(g["model"].unique())
    cats = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
    fig, ax = plt.subplots(figsize=(10, 5))
    import numpy as np

    width = 0.8 / max(1, len(models))
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        vals = [g[(g.model == m) & (g.category == c)]["pct_high"].mean()
                for c in cats]
        vals = [0 if (v != v) else v for v in vals]  # NaN -> 0
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20)
    ax.set_ylabel("% responses scoring >=5")
    ax.set_title("Frustration by evaluation category")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig3_per_turn(df, out_path, category="extended"):
    """Per-turn mean frustration with 95% CIs (Figure 3)."""
    pt = analysis.per_turn_summary(df, categories=(category,))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for m, grp in pt.groupby("model"):
        grp = grp.sort_values("turn")
        ax1.plot(grp["turn"], grp["mean_rating"], marker="o", label=m)
        ax1.fill_between(grp["turn"], grp["ci_lo"], grp["ci_hi"], alpha=0.15)
        ax2.plot(grp["turn"], grp["pct_high"], marker="o", label=m)
    ax1.set(xlabel="Turn", ylabel="Mean frustration",
            title=f"Mean frustration per turn ({category})")
    ax2.set(xlabel="Turn", ylabel="% scoring >=5",
            title=f"% high frustration per turn ({category})")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
