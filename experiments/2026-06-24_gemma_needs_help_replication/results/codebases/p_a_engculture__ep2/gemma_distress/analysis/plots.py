"""Matplotlib figures reproducing Figures 1, 2, and 3.

These are convenience plotters over the metrics computed in :mod:`metrics` and
:mod:`per_turn`. They save PNGs and never call ``plt.show`` so they work headless.
"""

from __future__ import annotations

from typing import Optional

from .metrics import CATEGORIES


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_figure1(ranking: list[tuple[str, float]], out_path: str) -> str:
    """Horizontal bar chart of avg %>=5 across categories, per model (Figure 1, left)."""
    plt = _import_plt()
    names = [n for n, _ in ranking][::-1]
    vals = [v for _, v in ranking][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1))
    ax.barh(names, vals, color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    for y, v in enumerate(vals):
        if v == v:
            ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title("Frustration across models")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure2(per_model: dict, out_path: str, threshold: int = 5) -> str:
    """Grouped bars: per-category mean (top) and %>=threshold (bottom) per model (Figure 2)."""
    plt = _import_plt()
    import numpy as np

    models = list(per_model.keys())
    cats = [c for c in CATEGORIES]
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for mi, model in enumerate(models):
        bc = per_model[model]["by_category"]
        means = [bc.get(c, {}).get("mean", float("nan")) for c in cats]
        pcts = [bc.get(c, {}).get("pct_high", float("nan")) for c in cats]
        offset = (mi - (len(models) - 1) / 2) * width
        ax_mean.bar(x + offset, means, width, label=model)
        ax_pct.bar(x + offset, pcts, width, label=model)
    ax_mean.set_ylabel("Mean frustration")
    ax_pct.set_ylabel(f"% scores >= {threshold}")
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(cats, rotation=30, ha="right")
    ax_mean.legend(fontsize=8)
    ax_mean.set_title("Frustration by category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure3(progressions: dict[str, dict], out_path: str) -> str:
    """Per-turn mean with 95% CI band, one line per model (Figure 3).

    ``progressions`` maps model name -> output of ``per_turn_progression`` for a category.
    """
    plt = _import_plt()
    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(12, 4))
    for model, prog in progressions.items():
        turns = prog["turns"]
        ax_mean.plot(turns, prog["mean"], marker="o", label=model)
        ax_mean.fill_between(turns, prog["mean_lo"], prog["mean_hi"], alpha=0.2)
        ax_pct.plot(turns, prog["pct_high"], marker="o", label=model)
        ax_pct.fill_between(turns, prog["pct_lo"], prog["pct_hi"], alpha=0.2)
    ax_mean.set_xlabel("Turn")
    ax_mean.set_ylabel("Mean frustration")
    ax_pct.set_xlabel("Turn")
    ax_pct.set_ylabel("% scores >= 5")
    ax_mean.legend(fontsize=8)
    ax_mean.set_title("Per-turn mean")
    ax_pct.set_title("Per-turn % high")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
