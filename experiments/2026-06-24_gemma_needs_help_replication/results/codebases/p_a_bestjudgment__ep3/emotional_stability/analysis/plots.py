"""Plotting for Figures 1-3, 5-8 (mean score + %>=5 bars and per-turn curves).

Matplotlib only; no seaborn. Each function takes already-aggregated stats and
writes a PNG. Kept deliberately simple — these mirror the paper's figure shapes
rather than its exact styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..utils.io import ensure_dir
from .metrics import CategoryStats, TurnPoint


def plot_model_comparison_bars(
    stats_by_model: dict[str, CategoryStats],
    out_path: str | Path,
    *,
    title: str = "Average % high-frustration responses",
    metric: str = "pct_high",
) -> None:
    """Figure 1 / Figure 5 style: one bar per model, sorted descending."""
    items = sorted(stats_by_model.items(), key=lambda kv: getattr(kv[1], metric),
                   reverse=True)
    labels = [k for k, _ in items]
    values = [getattr(v, metric) for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("%  responses scoring >= 5" if metric == "pct_high" else "mean score")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1f}", va="center")
    _save(fig, out_path)


def plot_category_grid(
    stats_by_model: dict[str, dict[str, CategoryStats]],
    out_path: str | Path,
    *,
    metric: str = "pct_high",
) -> None:
    """Figure 2: grouped bars, models x categories."""
    models = list(stats_by_model)
    categories = sorted({c for s in stats_by_model.values() for c in s})

    fig, ax = plt.subplots(figsize=(11, 5))
    n = len(models)
    width = 0.8 / max(1, n)
    x = range(len(categories))
    for i, m in enumerate(models):
        vals = [getattr(stats_by_model[m].get(c), metric, float("nan"))
                if stats_by_model[m].get(c) else float("nan")
                for c in categories]
        ax.bar([xi + i * width for xi in x], vals, width=width, label=m)
    ax.set_xticks([xi + width * (n - 1) / 2 for xi in x])
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("% scoring >= 5" if metric == "pct_high" else "mean score")
    ax.legend(fontsize=8)
    ax.set_title("Frustration by category and model")
    _save(fig, out_path)


def plot_per_turn(
    curves_by_model: dict[str, list[TurnPoint]],
    out_path: str | Path,
    *,
    metric: str = "mean_score",
) -> None:
    """Figure 3 / 8: per-turn progression with shaded 95% CIs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, pts in curves_by_model.items():
        turns = [p.turn for p in pts]
        if metric == "mean_score":
            ys = [p.mean_score for p in pts]
            los = [p.mean_ci[0] for p in pts]
            his = [p.mean_ci[1] for p in pts]
        else:
            ys = [p.pct_high for p in pts]
            los = [p.pct_high_ci[0] for p in pts]
            his = [p.pct_high_ci[1] for p in pts]
        line, = ax.plot(turns, ys, marker="o", label=model)
        ax.fill_between(turns, los, his, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Turn")
    ax.set_ylabel("mean frustration" if metric == "mean_score" else "% scoring >= 5")
    ax.legend(fontsize=8)
    ax.set_title("Per-turn frustration")
    _save(fig, out_path)


def _save(fig, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
