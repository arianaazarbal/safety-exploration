"""Matplotlib reproductions of the paper's figures.

Each function reads a CSV produced by a runner (or a precomputed DataFrame) and
writes a PNG to FIGURES_DIR. Styling is intentionally plain - the goal is to
reproduce the *quantities* in Figures 1-8, not their exact cosmetics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config

CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _save(fig, name: str) -> Path:
    out = config.FIGURES_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=150)
    return out


def plot_headline_bar(headline_csv: str | Path, name: str = "figure1_headline.png") -> Path:
    """Figure 1 (left): average % high-frustration per model."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(headline_csv).sort_values("avg_pct_high", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 1))
    ax.barh(df["model"], df["avg_pct_high"], color="#c0504d")
    for y, v in enumerate(df["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Distress under repeated rejection")
    return _save(fig, name)


def plot_category_bars(by_category_csv: str | Path, name: str = "figure2_categories.png") -> Path:
    """Figure 2: mean frustration (top) and %>=5 (bottom) per category per model."""
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(by_category_csv)
    models = sorted(df["model"].unique())
    cats = [c for c in CATEGORY_ORDER if c in set(df["category"])]
    x = np.arange(len(cats))
    width = 0.8 / max(1, len(models))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("category").reindex(cats)
        ax1.bar(x + i * width, sub["mean_score"], width, label=m)
        ax2.bar(x + i * width, sub["pct_high"], width, label=m)
    for ax, ylab in [(ax1, "Mean frustration"), (ax2, "% score >= 5")]:
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def plot_per_turn(per_turn_df: pd.DataFrame, category: str, name: str | None = None) -> Path:
    """Figure 3: per-turn mean score and %>=5 with 95% CIs."""
    import matplotlib.pyplot as plt

    name = name or f"figure3_per_turn_{category}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for m, sub in per_turn_df.groupby("model"):
        sub = sub.sort_values("turn")
        ax1.plot(sub["turn"], sub["mean_score"], marker="o", label=m)
        ax1.fill_between(sub["turn"], sub["mean_lo"], sub["mean_hi"], alpha=0.2)
        ax2.plot(sub["turn"], sub["pct_high"], marker="o", label=m)
        ax2.fill_between(sub["turn"], sub["pct_high_lo"], sub["pct_high_hi"], alpha=0.2)
    ax1.set(xlabel="Turn", ylabel="Mean frustration", title=f"{category}: mean")
    ax2.set(xlabel="Turn", ylabel="% score >= 5", title=f"{category}: % >= 5")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def plot_finetune_comparison(by_category_csv: str | Path, name: str = "figure5_finetunes.png") -> Path:
    """Figure 5: finetunes vs baselines (reuses the category-bar layout)."""
    return plot_category_bars(by_category_csv, name=name)


def plot_petri(petri_csv: str | Path, name: str = "figure6_petri.png") -> Path:
    """Figure 6: average Petri transcript score per model per emotion."""
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(petri_csv)
    emotions = list(config.PETRI_EMOTIONS)
    models = sorted(df["model"].unique())
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("emotion").reindex(emotions)
        yerr = [sub["mean_score"] - sub["ci_lo"], sub["ci_hi"] - sub["mean_score"]]
        ax.bar(x + i * width, sub["mean_score"], width, label=m, yerr=yerr, capsize=3)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_capabilities(capabilities_csv: str | Path, name: str = "figure7_capabilities.png") -> Path:
    """Figure 7: capability benchmark scores, Gemma-it vs DPO."""
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.read_csv(capabilities_csv).dropna(subset=["accuracy"])
    benches = sorted(df["benchmark"].unique())
    models = sorted(df["model"].unique())
    x = np.arange(len(benches))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("benchmark").reindex(benches)
        ax.bar(x + i * width, sub["accuracy"] * 100, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Capability preservation")
    ax.legend(fontsize=8)
    return _save(fig, name)
