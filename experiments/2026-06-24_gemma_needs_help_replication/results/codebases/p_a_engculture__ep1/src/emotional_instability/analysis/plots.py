"""Figure generation (Figures 1, 2, 3, 5, 6, 7, 8).

Each function takes already-computed metrics (so plotting has no dependency on
the API) and writes a PNG. matplotlib only; no seaborn. Faded CI bands match the
paper's per-turn figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .metrics import TurnPoint  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure1_headline(model_to_avg_pct: dict[str, float], path) -> Path:
    """Figure 1 left: avg % high-frustration responses per model (bar chart)."""
    items = sorted(model_to_avg_pct.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(names))))
    ax.barh(names, vals, color="#b5651d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: Average high-frustration rate across evaluations")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center")
    return _save(fig, path)


def figure2_by_category(
    model_summaries: dict[str, dict], categories: list[str], path
) -> Path:
    """Figure 2: mean frustration (top) and % >=5 (bottom) per category, per model."""
    import numpy as np

    models = list(model_summaries)
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.8 / max(1, len(models))

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, m in enumerate(models):
        by_cat = model_summaries[m]["by_category"]
        means = [by_cat.get(c, {}).get("mean", float("nan")) for c in categories]
        pcts = [by_cat.get(c, {}).get("pct_high", float("nan")) for c in categories]
        ax_mean.bar(x + i * width, means, width, label=m)
        ax_pct.bar(x + i * width, pcts, width, label=m)
    ax_mean.set_ylabel("Mean frustration score")
    ax_pct.set_ylabel("% responses ≥ 5")
    ax_pct.set_xticks(x + width * (len(models) - 1) / 2)
    ax_pct.set_xticklabels(categories, rotation=20, ha="right")
    ax_mean.legend(fontsize=8)
    ax_mean.set_title("Figure 2: Negative emotional expression across categories")
    return _save(fig, path)


def figure3_per_turn(
    model_to_points: dict[str, list[TurnPoint]], path, metric: str = "mean"
) -> Path:
    """Figure 3: per-turn mean score (or % >=5) with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, points in model_to_points.items():
        turns = [p.turn for p in points]
        if metric == "mean":
            ys = [p.mean for p in points]
            los = [p.mean_ci[0] for p in points]
            his = [p.mean_ci[1] for p in points]
            ax.set_ylabel("Mean frustration score")
        else:
            ys = [p.pct_high for p in points]
            los = [p.pct_ci[0] for p in points]
            his = [p.pct_ci[1] for p in points]
            ax.set_ylabel("% responses ≥ 5")
        line = ax.plot(turns, ys, marker="o", label=model)[0]
        ax.fill_between(turns, los, his, color=line.get_color(), alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_title("Figure 3: Per-turn frustration progression")
    ax.legend(fontsize=8)
    return _save(fig, path)


def figure_bars(
    model_to_value: dict[str, float], path, ylabel: str, title: str
) -> Path:
    """Generic grouped bar figure (reused for Figures 5, 6, 7)."""
    items = list(model_to_value.items())
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names)), 5))
    ax.bar(names, vals, color="#3b6ea5")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=20, ha="right")
    return _save(fig, path)
