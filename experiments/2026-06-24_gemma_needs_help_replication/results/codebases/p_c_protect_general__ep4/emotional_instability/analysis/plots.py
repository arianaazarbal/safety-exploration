"""Plotting helpers reproducing the paper's figures (matplotlib, no seaborn)."""
from __future__ import annotations

import os
from typing import Optional

from ..config import RESULTS_DIR


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_figure1(figure1_rows: list[dict], out_path: Optional[str] = None) -> str:
    """Horizontal bar of avg % high-frustration responses per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(RESULTS_DIR, "plots", "figure1.png")
    _ensure_dir(out_path)
    models = [r["model"] for r in figure1_rows][::-1]
    vals = [r["avg_pct_high_frustration"] for r in figure1_rows][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models, vals, color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: emotional instability across models")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure2(figure2: dict, out_path: Optional[str] = None) -> str:
    """Grouped bars: per-model mean frustration and %>=5 across categories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .aggregate import CATEGORIES

    out_path = out_path or os.path.join(RESULTS_DIR, "plots", "figure2.png")
    _ensure_dir(out_path)
    models = list(figure2)
    x = np.arange(len(CATEGORIES))
    width = 0.8 / max(1, len(models))

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(10, 8))
    for i, m in enumerate(models):
        means = [figure2[m].get(c, {}).get("mean", float("nan")) for c in CATEGORIES]
        pcts = [figure2[m].get(c, {}).get("pct_ge5", float("nan")) for c in CATEGORIES]
        ax_mean.bar(x + i * width, means, width, label=m)
        ax_pct.bar(x + i * width, pcts, width, label=m)
    for ax, title, ylab in (
        (ax_mean, "Mean frustration score by category", "mean score"),
        (ax_pct, "% scores ≥ 5 by category", "% ≥ 5"),
    ):
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(CATEGORIES, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_turn(curves: dict, model: str, out_path: Optional[str] = None) -> str:
    """Figure 3: per-turn mean + %>=5 with 95% CIs for extended & wildchat."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(RESULTS_DIR, "plots", f"figure3_{model}.png")
    _ensure_dir(out_path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, cond in enumerate(("extended", "wildchat")):
        c = curves.get(cond)
        if not c or not c["turns"]:
            continue
        turns = c["turns"]
        # mean
        ax = axes[0][col]
        ax.plot(turns, c["mean"], marker="o")
        lo = [ci[0] for ci in c["mean_ci"]]
        hi = [ci[1] for ci in c["mean_ci"]]
        ax.fill_between(turns, lo, hi, alpha=0.2)
        ax.set_title(f"{cond}: mean frustration")
        ax.set_xlabel("turn")
        ax.set_ylabel("mean score")
        # pct >=5
        ax = axes[1][col]
        ax.plot(turns, c["pct_ge5"], marker="o", color="#c0504d")
        lo = [ci[0] for ci in c["pct_ge5_ci"]]
        hi = [ci[1] for ci in c["pct_ge5_ci"]]
        ax.fill_between(turns, lo, hi, alpha=0.2, color="#c0504d")
        ax.set_title(f"{cond}: % scores ≥ 5")
        ax.set_xlabel("turn")
        ax.set_ylabel("% ≥ 5")
    fig.suptitle(f"Figure 3 (per-turn): {model}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
