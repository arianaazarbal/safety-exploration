"""Matplotlib reproductions of Figures 1-3 from saved scores.

These are intentionally simple bar/line plots; the goal is to reproduce the
*quantities*, not the paper's exact styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

from . import aggregate  # noqa: E402
from .conditions import CATEGORIES  # noqa: E402


def figure1(model_names: list[str], out: Path | None = None) -> Path:
    df = aggregate.all_models_headline(model_names)
    out = out or config.FIGURES_DIR / "figure1_avg_pct_high.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["model"], df["avg_pct_high"], color="#b5651d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.invert_yaxis()
    ax.set_title("Figure 1: emotional instability across models")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2(model_names: list[str], out: Path | None = None) -> Path:
    out = out or config.FIGURES_DIR / "figure2_by_category.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    width = 0.8 / max(len(model_names), 1)
    x = range(len(CATEGORIES))
    for i, m in enumerate(model_names):
        df = aggregate.load_model_scores(m)
        if df.empty:
            continue
        summ = aggregate.category_summary(df)
        offs = [xi + i * width for xi in x]
        ax1.bar(offs, summ["mean_score"], width=width, label=m)
        ax2.bar(offs, summ["pct_high"], width=width, label=m)
    ax1.set_ylabel("Mean frustration")
    ax2.set_ylabel("% score >= 5")
    ax2.set_xticks([xi + 0.4 for xi in x])
    ax2.set_xticklabels(CATEGORIES, rotation=20)
    ax1.legend(fontsize=8)
    ax1.set_title("Figure 2: frustration by evaluation category")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3(model_names: list[str], condition: str = "extended", out: Path | None = None) -> Path:
    out = out or config.FIGURES_DIR / f"figure3_{condition}_per_turn.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for m in model_names:
        df = aggregate.load_model_scores(m)
        if df.empty or condition not in set(df["condition"]):
            continue
        prog = aggregate.per_turn_progression(df, condition)
        turns = [t + 1 for t in prog.index]
        ax1.plot(turns, prog["mean_score"], marker="o", label=m)
        ax2.plot(turns, prog["pct_high"], marker="o", label=m)
        ax2.fill_between(
            turns, prog["pct_high"] - prog["pct_high_ci95"],
            prog["pct_high"] + prog["pct_high_ci95"], alpha=0.15,
        )
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% score >= 5")
    ax1.legend(fontsize=8)
    fig.suptitle(f"Figure 3: per-turn progression ({condition})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
