"""Plotting helpers reproducing Figures 1-3, 5-8.

These are intentionally thin wrappers over matplotlib: each takes already-aggregated
frames (from :mod:`aggregate`) and writes a PNG. Kept dependency-light and headless
(``Agg`` backend) so figures can be produced on a server.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_model_comparison(summaries, out: Path, *, metric: str = "pct_high") -> Path:
    """Figure 1/2 bar chart: per-model avg % high-frustration (or mean)."""
    summaries = sorted(summaries, key=lambda s: getattr(s, "pct_high"), reverse=True)
    labels = [s.model for s in summaries]
    if metric == "pct_high":
        vals = [s.pct_high for s in summaries]
        ylabel = "% responses scoring ≥5"
    else:
        vals = [s.mean_frustration for s in summaries]
        ylabel = "mean frustration"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, vals, color="#c0504d")
    ax.set_ylabel(ylabel)
    ax.set_title("Negative emotional expression across models")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_turn(curve_df, out: Path, *, metric: str = "mean") -> Path:
    """Figure 3: per-turn progression with 95% CI bands, one line per condition."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ycol, cicol = (("mean", "mean_ci") if metric == "mean" else ("pct_high", "pct_high_ci"))
    for cond, grp in curve_df.groupby("condition"):
        grp = grp.sort_values("turn")
        ax.plot(grp["turn"], grp[ycol], marker="o", label=cond)
        ax.fill_between(grp["turn"], grp[ycol] - grp[cicol], grp[ycol] + grp[cicol], alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_ylabel("mean frustration" if metric == "mean" else "% scoring ≥5")
    ax.set_title("Per-turn frustration progression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_grouped_bars(data: dict[str, dict[str, float]], out: Path, *, ylabel: str,
                      title: str) -> Path:
    """Generic grouped bars: {group: {series: value}} -> clustered bar chart.

    Used for Petri (Figure 6: emotion category x model) and capability/intervention
    comparisons (Figures 5, 7)."""
    import numpy as np

    groups = list(data.keys())
    series = sorted({s for g in data.values() for s in g})
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, s in enumerate(series):
        vals = [data[g].get(s, 0.0) for g in groups]
        ax.bar(x + i * width, vals, width, label=s)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
