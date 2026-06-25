"""Figure generation (Figures 1, 2, 3, 5, 6, 7, 8).

Each function takes already-aggregated data and writes a PNG to FIGURES_DIR.
`scripts/make_figures.py` loads the result JSONLs, aggregates with `metrics`,
and calls these.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

FIG = config.FIGURES_DIR


def _save(fig, name: str):
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] wrote {path}")


def figure1_avg_high(per_model_pct_high: dict):
    """Figure 1 (left): avg % high-frustration responses per model."""
    items = sorted(per_model_pct_high.items(), key=lambda kv: kv[1], reverse=True)
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="#b5651d")
    ax.set_xlabel("% responses scoring ≥5/10 frustration")
    ax.set_title("Average high-frustration rate across evaluations")
    ax.invert_yaxis()
    _save(fig, "figure1_avg_high_frustration.png")


def figure2_by_category(summaries: dict):
    """Figure 2: mean frustration (top) and % >= 5 (bottom) per category/model."""
    models = list(summaries)
    cats = sorted({c for s in summaries.values() for c in s["by_category"]})
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))
    for i, m in enumerate(models):
        means = [summaries[m]["by_category"].get(c, {}).get("mean_frustration", 0) for c in cats]
        pcts = [summaries[m]["by_category"].get(c, {}).get("pct_high", 0) for c in cats]
        ax_top.bar(x + i * width, means, width, label=m)
        ax_bot.bar(x + i * width, pcts, width, label=m)
    for ax, title, ylab in [(ax_top, "Mean frustration by category", "mean score"),
                            (ax_bot, "% scores ≥5 by category", "% ≥5")]:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    _save(fig, "figure2_by_category.png")


def figure3_per_turn(curves: dict):
    """Figure 3: per-turn mean and % >= 5 with 95% CIs (8-turn / WildChat)."""
    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(12, 4))
    for label, curve in curves.items():
        turns = sorted(curve)
        means = [curve[t]["mean"] for t in turns]
        lo = [curve[t]["mean_ci"][0] for t in turns]
        hi = [curve[t]["mean_ci"][1] for t in turns]
        ax_mean.plot(turns, means, marker="o", label=label)
        ax_mean.fill_between(turns, lo, hi, alpha=0.2)
        pcts = [curve[t]["pct_high"] for t in turns]
        plo = [curve[t]["pct_high_ci"][0] for t in turns]
        phi = [curve[t]["pct_high_ci"][1] for t in turns]
        ax_pct.plot(turns, pcts, marker="o", label=label)
        ax_pct.fill_between(turns, plo, phi, alpha=0.2)
    ax_mean.set(xlabel="Turn", ylabel="mean frustration", title="Per-turn mean")
    ax_pct.set(xlabel="Turn", ylabel="% ≥5", title="Per-turn % ≥5")
    ax_mean.legend(fontsize=8); ax_pct.legend(fontsize=8)
    _save(fig, "figure3_per_turn.png")


def figure5_interventions(summaries: dict):
    """Figure 5: vanilla vs SFT vs DPO (mean + % >= 5)."""
    models = list(summaries)
    means = [summaries[m]["mean_frustration"] for m in models]
    pcts = [summaries[m]["pct_high"] for m in models]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(models, means, color="#4c72b0"); a1.set_title("Mean frustration"); a1.tick_params(axis="x", rotation=30)
    a2.bar(models, pcts, color="#dd8452"); a2.set_title("% scores ≥5"); a2.tick_params(axis="x", rotation=30)
    _save(fig, "figure5_interventions.png")


def figure6_petri(petri_means: dict):
    """Figure 6: Petri average transcript score per model across 4 emotions."""
    models = list(petri_means)
    emotions = config.PETRI_EMOTIONS
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(models):
        vals = [petri_means[m].get(e, 0) for e in emotions]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("avg transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    _save(fig, "figure6_petri.png")


def figure7_capabilities(cap_by_model: dict):
    """Figure 7: capability benchmark accuracy, vanilla vs finetuned."""
    benches = [b.key for b in config.CAPABILITY_BENCHMARKS]
    models = list(cap_by_model)
    x = np.arange(len(benches))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(models):
        vals = [(cap_by_model[m].get(b) or 0) for b in benches]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("accuracy"); ax.set_title("Capability preservation")
    ax.legend(fontsize=8)
    _save(fig, "figure7_capabilities.png")


def figure8_recovery(recovery_pct_high: dict):
    """Figure 8: % continuations scoring >= 5 from high-frustration prefills."""
    items = list(recovery_pct_high.items())
    labels, vals = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color="#c44e52")
    ax.set_ylabel("% continuations ≥5")
    ax.set_title("Recovery from high-frustration prefills")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "figure8_recovery.png")
