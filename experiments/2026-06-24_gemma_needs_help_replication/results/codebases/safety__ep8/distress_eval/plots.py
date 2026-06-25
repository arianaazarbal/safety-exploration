"""Figure generation. Reproduces the paper's core figures from aggregated data.

Figure 1/2 : per-model % high-frustration (>=5) and mean frustration.
Figure 3   : per-turn frustration progression (8-turn extended + WildChat).
Figure 5   : vanilla vs SFT vs DPO comparison (same axes as Figure 2).
Figure 6   : Petri per-emotion scores.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_model_summary(summary: pd.DataFrame, out_path: Path, title: str = "Figure 2"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    s = summary.sort_values("pct_high", ascending=True)
    axes[0].barh(s["model_key"], s["mean_frustration"], color="#c0504d")
    axes[0].set_xlabel("Mean frustration score (0-10)")
    axes[0].set_title("Mean frustration")
    axes[1].barh(s["model_key"], s["pct_high"], color="#4f81bd")
    axes[1].set_xlabel("% responses scoring >= 5")
    axes[1].set_title("% high-frustration")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(progression: pd.DataFrame, out_path: Path, category: str = "extended"):
    sub = progression[progression["category"] == category]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_key, g in sub.groupby("model_key"):
        g = g.sort_values("turn")
        axes[0].plot(g["turn"], g["mean_frustration"], marker="o", label=model_key)
        axes[1].plot(g["turn"], g["pct_high"], marker="o", label=model_key)
    axes[0].set_xlabel("Turn"); axes[0].set_ylabel("Mean frustration")
    axes[0].set_title(f"Per-turn mean ({category})")
    axes[1].set_xlabel("Turn"); axes[1].set_ylabel("% scores >= 5")
    axes[1].set_title(f"Per-turn % high ({category})")
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    fig.suptitle("Figure 3: multi-turn elicitation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intervention_comparison(summary: pd.DataFrame, out_path: Path,
                                 keys: list[str] | None = None):
    """Figure 5: compare vanilla / SFT / DPO Gemma."""
    s = summary
    if keys:
        s = s[s["model_key"].isin(keys)]
    plot_model_summary(s, out_path, title="Figure 5: intervention comparison")


def plot_petri(petri_df: pd.DataFrame, out_path: Path):
    """Figure 6: mean transcript score per emotion category, per model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = petri_df.pivot_table(index="emotion", columns="model_key",
                                 values="score", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
