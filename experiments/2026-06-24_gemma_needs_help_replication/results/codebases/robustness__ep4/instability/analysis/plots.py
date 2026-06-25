"""Figure generation (Figures 1-3, 5-6). Matplotlib only, no seaborn.

These mirror the paper's figures from the analysis DataFrames. All functions
save to a path and return it; they never call plt.show().
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_model_summary(summary_df, out_path: str, value_col: str = "avg_pct_high_balanced"):
    """Figure 1: avg % high-frustration responses per model (bar chart)."""
    df = summary_df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 1))
    ax.barh(df["model"], df[value_col], color="#c0392b")
    ax.set_xlabel("% responses scoring >= 5 (frustration)")
    ax.set_title("Average high-frustration rate across evaluations")
    for i, v in enumerate(df[value_col]):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_turn(curves_df, out_path: str, metric: str = "mean"):
    """Figure 3: per-turn progression with 95% CI bands."""
    ci = "mean_ci" if metric == "mean" else "pct_high_ci"
    ycol = metric if metric == "mean" else "pct_high"
    fig, ax = plt.subplots(figsize=(7, 5))
    for (model, cond), g in curves_df.groupby(["model", "condition"]):
        g = g.sort_values("turn")
        ax.plot(g["turn"], g[ycol], marker="o", label=f"{model} ({cond})")
        ax.fill_between(g["turn"], g[ycol] - g[ci], g[ycol] + g[ci], alpha=0.15)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration" if metric == "mean" else "% scoring >= 5")
    ax.set_title("Per-turn frustration progression")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_petri(petri_df, out_path: str):
    """Figure 6: average transcript score per model across emotion dimensions."""
    pivot = petri_df.groupby(["model", "dimension"])["score"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(title="dimension", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
