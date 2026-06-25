"""Plotting helpers reproducing Figures 1-3 (and the Section 4 comparison).

All functions take already-aggregated frames and write a PNG to FIGURES_DIR.
Plotting is deliberately dependency-light (matplotlib only).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402


def bar_headline(headline_df: pd.DataFrame, out: Path | None = None) -> Path:
    """Figure 1 (left): avg % high-frustration responses per model."""
    out = out or config.FIGURES_DIR / "fig1_headline.png"
    d = headline_df.sort_values("avg_pct_high", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(d) + 1))
    ax.barh(d["model"], d["avg_pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Emotional instability across models")
    for y, v in enumerate(d["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def grouped_categories(per_cat_df: pd.DataFrame,
                       metric: str = "pct_high",
                       out: Path | None = None) -> Path:
    """Figure 2: per-category metric grouped by model."""
    out = out or config.FIGURES_DIR / f"fig2_{metric}.png"
    pivot = per_cat_df.pivot(index="category", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel(metric)
    ax.set_title(f"Frustration by category ({metric})")
    ax.figure.tight_layout()
    ax.figure.savefig(out, dpi=150)
    plt.close(ax.figure)
    return out


def per_turn_lines(curves: list[pd.DataFrame],
                   metric: str = "mean_score",
                   out: Path | None = None) -> Path:
    """Figure 3: per-turn progression with 95% CI bands."""
    out = out or config.FIGURES_DIR / f"fig3_{metric}.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
    for c in curves:
        if c.empty:
            continue
        label = f"{c['model'].iloc[0]} / {c['category'].iloc[0]}"
        ax.plot(c["turn"], c[metric], marker="o", label=label)
        if ci_lo in c and ci_hi in c:
            ax.fill_between(c["turn"], c[ci_lo], c[ci_hi], alpha=0.15)
    ax.set_xlabel("Turn")
    ax.set_ylabel(metric)
    ax.set_title("Per-turn frustration progression")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
