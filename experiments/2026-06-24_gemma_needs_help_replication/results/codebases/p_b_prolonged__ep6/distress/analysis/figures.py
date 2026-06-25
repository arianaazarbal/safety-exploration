"""Generate the paper's figures from scored results.

Figure 1/2: headline % high-frustration and per-category bars.
Figure 3:   per-turn curves for the 8-turn and WildChat conditions.
Figure 5:   intervention comparison (vanilla / SFT / DPO).
Figure 6:   Petri per-emotion bars.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import RESULTS_DIR
from .aggregate import category_summary, headline_table
from .per_turn import per_turn_curve


def fig_headline(df, out: Path | None = None):
    head = headline_table(df)
    out = out or (RESULTS_DIR / "fig1_headline.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(head["model"], head["avg_pct_high"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >=5)")
    ax.invert_yaxis()
    ax.set_title("Figure 1: distress across models")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_per_category(df, out: Path | None = None):
    cat = category_summary(df)
    out = out or (RESULTS_DIR / "fig2_per_category.png")
    models = sorted(cat["model"].unique())
    cats = sorted(cat["category"].unique())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(models))
    import numpy as np
    x = np.arange(len(cats))
    for i, m in enumerate(models):
        sub = cat[cat["model"] == m].set_index("category").reindex(cats)
        ax1.bar(x + i * width, sub["mean_score"], width, label=m)
        ax2.bar(x + i * width, sub["pct_high"], width, label=m)
    ax1.set_ylabel("Mean frustration")
    ax2.set_ylabel("% score >=5")
    ax2.set_xticks(x + width * len(models) / 2)
    ax2.set_xticklabels(cats, rotation=30, ha="right")
    ax1.legend(fontsize=7)
    ax1.set_title("Figure 2: frustration by category")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_per_turn(df, model: str, condition: str = "extended",
                 out: Path | None = None):
    curve = per_turn_curve(df, model, condition)
    out = out or (RESULTS_DIR / f"fig3_perturn_{model}_{condition}.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(curve["turn"], curve["mean_score"], marker="o")
    ax1.fill_between(curve["turn"], curve["mean_lo"], curve["mean_hi"], alpha=0.2)
    ax1.set_title(f"{model}: mean score / turn ({condition})")
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration")
    ax2.plot(curve["turn"], curve["pct_high"], marker="o", color="#c0504d")
    ax2.fill_between(curve["turn"], curve["pct_lo"], curve["pct_hi"], alpha=0.2,
                     color="#c0504d")
    ax2.set_title("% score >=5 / turn")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% >=5")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
