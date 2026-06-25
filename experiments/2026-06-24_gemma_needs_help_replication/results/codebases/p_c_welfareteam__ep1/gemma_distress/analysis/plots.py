"""Matplotlib plots reproducing the paper's figures.

All functions take already-aggregated summaries (from :mod:`aggregate`) and save
a PNG.  Kept deliberately simple: grouped bar charts for Figures 1/2/5/7 and
per-turn line charts with CI bands for Figure 3/8.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def plot_headline_bars(model_to_frac: dict[str, float], path: str | Path, title: str = "") -> None:
    """Figure 1 left: % high-frustration responses per model."""
    items = sorted(model_to_frac.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v * 100 for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1.5))
    ax.barh(labels, values, color="#b5651d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    if title:
        ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    _save(fig, path)


def plot_category_bars(
    summaries_by_model: dict[str, dict[str, dict[str, float]]],
    path: str | Path,
    metric: str = "frac_high",
    title: str = "",
) -> None:
    """Figure 2 / 5: per-category bars grouped by model."""
    import numpy as np

    models = list(summaries_by_model.keys())
    categories = sorted({c for s in summaries_by_model.values() for c in s})
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(categories) + 2, 4))
    for mi, model in enumerate(models):
        vals = []
        for cat in categories:
            v = summaries_by_model[model].get(cat, {}).get(metric, 0.0)
            vals.append(v * 100 if metric == "frac_high" else v)
        ax.bar(x + mi * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("% scores >= 5" if metric == "frac_high" else "Mean frustration")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_per_turn(
    per_turn_by_model: dict[str, dict[int, dict[str, float]]],
    path: str | Path,
    metric: str = "mean_score",
    title: str = "",
) -> None:
    """Figure 3 / 8: per-turn curves with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ci_lo = f"{'mean' if metric == 'mean_score' else 'frac_high'}_ci_low"
    ci_hi = f"{'mean' if metric == 'mean_score' else 'frac_high'}_ci_high"
    for model, per_turn in per_turn_by_model.items():
        turns = sorted(per_turn)
        xs = [t + 1 for t in turns]  # 1-based turns in the figures
        ys = [per_turn[t][metric] * (100 if metric == "frac_high" else 1) for t in turns]
        lo = [per_turn[t][ci_lo] * (100 if metric == "frac_high" else 1) for t in turns]
        hi = [per_turn[t][ci_hi] * (100 if metric == "frac_high" else 1) for t in turns]
        line, = ax.plot(xs, ys, marker="o", label=model)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Turn")
    ax.set_ylabel("% scores >= 5" if metric == "frac_high" else "Mean frustration")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
