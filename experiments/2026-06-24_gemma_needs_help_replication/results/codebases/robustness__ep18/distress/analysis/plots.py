"""Figure generation (Figures 1, 2, 3, 5, 6).

All plots read the persisted JSONL results and write PNGs under results/figures/.
Matplotlib only; no seaborn dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import RESULTS_DIR
from ..eval.metrics import headline, load_all, per_category, per_turn

FIG_DIR = RESULTS_DIR / "figures"


def _ensure():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig1_headline(model_names: list[str], threshold: int = 5) -> Path:
    """Figure 1 (left): avg % high-frustration responses per model."""
    _ensure()
    df = load_all(model_names)
    h = headline(df, threshold)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(h) + 1))
    ax.barh(h["model"], h["avg_pct_high_frustration"], color="#c0392b")
    ax.set_xlabel(f"Avg % responses with frustration >= {threshold}")
    ax.invert_yaxis()
    for i, v in enumerate(h["avg_pct_high_frustration"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    ax.set_title("Figure 1: High-frustration rate across evaluations")
    fig.tight_layout()
    out = FIG_DIR / "fig1_headline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig2_by_category(model_names: list[str], threshold: int = 5) -> Path:
    """Figure 2: mean frustration (top) and % >= threshold (bottom) per category."""
    _ensure()
    df = load_all(model_names)
    pc = per_category(df, threshold)
    categories = sorted(pc["category"].unique())
    models = list(dict.fromkeys(pc["model"]))
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for mi, m in enumerate(models):
        sub = pc[pc["model"] == m].set_index("category").reindex(categories)
        ax1.bar(x + mi * width, sub["mean_score"], width, label=m)
        ax2.bar(x + mi * width, sub["pct_high"], width, label=m)
    ax1.set_ylabel("Mean frustration score")
    ax2.set_ylabel(f"% responses >= {threshold}")
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(categories, rotation=20, ha="right")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_title("Figure 2: Frustration across evaluation categories")
    fig.tight_layout()
    out = FIG_DIR / "fig2_by_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig3_per_turn(model_names: list[str], condition: str = "extended", threshold: int = 5) -> Path:
    """Figure 3: per-turn progression with 95% CIs for an 8-turn / WildChat eval."""
    _ensure()
    df = load_all(model_names)
    pt = per_turn(df, condition, threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in sorted(pt["model"].unique()):
        s = pt[pt["model"] == m]
        ax1.plot(s["turn"], s["mean_score"], marker="o", label=m)
        ax1.fill_between(s["turn"], s["mean_lo"], s["mean_hi"], alpha=0.15)
        ax2.plot(s["turn"], s["pct_high"], marker="o", label=m)
        ax2.fill_between(s["turn"], s["pct_lo"], s["pct_hi"], alpha=0.15)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration score")
    ax2.set_xlabel("Turn"); ax2.set_ylabel(f"% >= {threshold}")
    ax1.legend(fontsize=8)
    fig.suptitle(f"Figure 3: Per-turn frustration ({condition})")
    fig.tight_layout()
    out = FIG_DIR / f"fig3_per_turn_{condition}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig5_intervention(model_names: list[str], threshold: int = 5) -> Path:
    """Figure 5: vanilla vs SFT vs DPO Gemma across the Section 2 evaluations.

    Pass the relevant model names (e.g. gemma-3-27b-it, gemma-3-27b-it-sft,
    gemma-3-27b-it-dpo); reuses the headline metric.
    """
    _ensure()
    df = load_all(model_names)
    h = headline(df, threshold)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(h["model"], h["mean_score"], color="#2980b9")
    ax1.set_ylabel("Mean frustration score")
    ax2.bar(h["model"], h["avg_pct_high_frustration"], color="#c0392b")
    ax2.set_ylabel(f"Avg % >= {threshold}")
    for ax in (ax1, ax2):
        ax.set_xticklabels(h["model"], rotation=25, ha="right", fontsize=8)
    fig.suptitle("Figure 5: Effect of SFT / DPO interventions (Gemma)")
    fig.tight_layout()
    out = FIG_DIR / "fig5_intervention.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig6_openended(model_names: list[str]) -> Path:
    """Figure 6: average open-ended (Petri) transcript score per emotion per model."""
    _ensure()
    rows = []
    for m in model_names:
        fp = RESULTS_DIR / "openended" / f"{m}.jsonl"
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("No open-ended results found.")
    df = pd.DataFrame(rows)
    piv = df.groupby(["model", "emotion"])["score"].mean().unstack("emotion")
    emotions = list(piv.columns)
    models = list(piv.index)
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, m in enumerate(models):
        ax.bar(x + mi * width, piv.loc[m].values, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.legend(fontsize=8)
    ax.set_title("Figure 6: Open-ended (Petri) emotion elicitation")
    fig.tight_layout()
    out = FIG_DIR / "fig6_openended.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
