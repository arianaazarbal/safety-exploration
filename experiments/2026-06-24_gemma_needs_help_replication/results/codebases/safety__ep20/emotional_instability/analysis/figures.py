"""Reproduce the paper's key figures from collected records.

* fig1_headline_bar     -> Figure 1 (avg % high-frustration per model)
* fig2_per_category     -> Figure 2 (mean score + %>=5 per category)
* fig3_per_turn         -> Figure 3 (per-turn progression with 95% CIs)
* fig5_intervention_bar -> Figure 5 (vanilla vs SFT vs DPO)
* fig6_petri            -> Figure 6 (Petri scores per emotion)

All functions take a DataFrame (see analysis.metrics) and save a PNG.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[figures] wrote {path}")
    return path


def fig1_headline_bar(df, path: str = "results/figures/fig1_headline.png"):
    series = metrics.headline_pct_high(df)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(series))))
    ax.barh(series.index[::-1], series.values[::-1], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: emotional instability across models")
    for i, v in enumerate(series.values[::-1]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)
    return _save(fig, path)


def fig2_per_category(df, path: str = "results/figures/fig2_per_category.png"):
    pc = metrics.per_category(df)
    models = sorted(pc["model"].unique())
    cats = list(pc["category"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(models))
    for mi, model in enumerate(models):
        sub = pc[pc["model"] == model].set_index("category").reindex(cats)
        xs = [c + mi * width for c in range(len(cats))]
        axes[0].bar(xs, sub["mean_frustration"], width=width, label=model)
        axes[1].bar(xs, sub["pct_high"] * 100, width=width, label=model)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score ≥ 5")
    axes[1].set_xticks([c + 0.4 for c in range(len(cats))])
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].set_title("Figure 2: frustration by evaluation category")
    axes[0].legend(fontsize=8)
    return _save(fig, path)


def fig3_per_turn(df, category: str = "extended",
                  path: str = "results/figures/fig3_per_turn.png"):
    pt = metrics.per_turn(df, category=category)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model in sorted(pt["model"].unique()):
        sub = pt[pt["model"] == model]
        axes[0].plot(sub["turn"], sub["mean"], marker="o", label=model)
        axes[0].fill_between(sub["turn"], sub["mean_lo"], sub["mean_hi"], alpha=0.15)
        axes[1].plot(sub["turn"], sub["pct_high"], marker="o", label=model)
        axes[1].fill_between(sub["turn"], sub["pct_lo"], sub["pct_hi"], alpha=0.15)
    axes[0].set(xlabel="Turn", ylabel="Mean frustration")
    axes[1].set(xlabel="Turn", ylabel="% score ≥ 5")
    axes[0].set_title(f"Figure 3: per-turn frustration ({category})")
    axes[0].legend(fontsize=8)
    return _save(fig, path)


def fig5_intervention_bar(df, path: str = "results/figures/fig5_intervention.png"):
    """Expects a df whose 'model' includes vanilla/SFT/DPO variants."""
    series = metrics.headline_pct_high(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(series.index, series.values, color="#2e86c1")
    ax.set_ylabel("Avg % high-frustration (score ≥ 5)")
    ax.set_title("Figure 5: intervention comparison")
    for i, v in enumerate(series.values):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    return _save(fig, path)


def fig6_petri(petri_df, path: str = "results/figures/fig6_petri.png"):
    summ = metrics.petri_summary(petri_df)
    models = sorted(summ["model"].unique())
    emotions = sorted(summ["emotion"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(models))
    for mi, model in enumerate(models):
        sub = summ[summ["model"] == model].set_index("emotion").reindex(emotions)
        xs = [e + mi * width for e in range(len(emotions))]
        yerr = [sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]]
        ax.bar(xs, sub["mean"], width=width, label=model, yerr=yerr, capsize=3)
    ax.set_xticks([e + 0.4 for e in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    return _save(fig, path)
