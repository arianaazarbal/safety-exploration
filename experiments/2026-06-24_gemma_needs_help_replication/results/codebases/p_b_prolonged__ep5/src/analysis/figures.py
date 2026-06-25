"""Generate the paper's figures from aggregated data (saved as PNGs in results/).

Covers Figure 1 (table -> bar), Figure 2 (per-category mean + %>=5), Figure 3
(per-turn with CIs), Figure 5 (finetuning comparison), Figure 6 (Petri), and
Figure 7 (capabilities). All plotting is matplotlib-only and side-effect free
except for writing PNGs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..config import RESULTS_DIR


def bar_figure1(table: pd.DataFrame, out: Path = None):
    out = out or (RESULTS_DIR / "figure1_high_frustration.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(table["model"], table["avg_pct_high_frustration"])
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def category_figure2(data: pd.DataFrame, out: Path = None):
    out = out or (RESULTS_DIR / "figure2_by_category.png")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for model, grp in data.groupby("model"):
        a1.plot(grp["category"], grp["mean_score"], marker="o", label=model)
        a2.plot(grp["category"], grp["pct_high"], marker="o", label=model)
    a1.set_ylabel("Mean frustration"); a2.set_ylabel("% score >= 5")
    a1.set_title("Figure 2: frustration by evaluation category")
    a1.legend(fontsize=7); a2.set_xlabel("category")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def per_turn_figure3(per_turn: dict[str, pd.DataFrame], out: Path = None):
    """per_turn: {label: dataframe from aggregate.per_turn_data}."""
    out = out or (RESULTS_DIR / "figure3_per_turn.png")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, df in per_turn.items():
        a1.plot(df["turn"], df["mean_score"], marker="o", label=label)
        a1.fill_between(df["turn"], df["score_lo"], df["score_hi"], alpha=0.2)
        a2.plot(df["turn"], df["pct_high"], marker="o", label=label)
        a2.fill_between(df["turn"], df["pct_lo"], df["pct_hi"], alpha=0.2)
    a1.set_xlabel("turn"); a1.set_ylabel("mean frustration"); a1.legend()
    a2.set_xlabel("turn"); a2.set_ylabel("% score >= 5"); a2.legend()
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def finetuning_figure5(table: pd.DataFrame, out: Path = None):
    """Reuse the Figure-1 style bar for {vanilla, DPO, SFT-diverse, SFT-teacher}."""
    out = out or (RESULTS_DIR / "figure5_finetuning.png")
    return bar_figure1(table, out)


def petri_figure6(summary: pd.DataFrame, out: Path = None):
    out = out or (RESULTS_DIR / "figure6_petri.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    emotions = sorted(summary["emotion"].unique())
    models = sorted(summary["model"].unique())
    width = 0.8 / max(len(models), 1)
    import numpy as np
    x = np.arange(len(emotions))
    for i, model in enumerate(models):
        sub = summary[summary["model"] == model].set_index("emotion").reindex(emotions)
        ax.bar(x + i * width, sub["mean_score"], width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("mean transcript score"); ax.legend(fontsize=7)
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def capabilities_figure7(df: pd.DataFrame, out: Path = None):
    out = out or (RESULTS_DIR / "figure7_capabilities.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    benches = sorted(df["benchmark"].unique())
    models = sorted(df["model"].unique())
    import numpy as np
    x = np.arange(len(benches)); width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("benchmark").reindex(benches)
        ax.bar(x + i * width, sub["accuracy"], width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2); ax.set_xticklabels(benches)
    ax.set_ylabel("accuracy"); ax.legend(fontsize=7)
    ax.set_title("Figure 7: capability preservation after finetuning")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out
