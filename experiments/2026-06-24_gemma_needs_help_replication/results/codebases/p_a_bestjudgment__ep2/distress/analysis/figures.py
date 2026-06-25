"""Produce the paper's figures from scored data.

All functions take in-memory metric structures (from ``distress.metrics`` /
``distress.petri`` / ``distress.capabilities``) and write a PNG. Kept free of
side effects beyond saving the figure so they can be unit-driven from saved
JSONL.
"""

from __future__ import annotations

from ..judge import Score
from ..metrics import (
    Aggregate,
    by_model_category,
    by_model_overall,
    headline_pct_high,
    per_turn,
)

CATEGORY_ORDER = ["numeric", "triggers", "tones", "extended", "wildchat"]


def figure1_table(scores: list[Score]) -> list[tuple[str, float]]:
    """Figure 1's ranked 'Avg % high-frustration responses' table."""
    pct = headline_pct_high(scores)
    return sorted(pct.items(), key=lambda kv: kv[1], reverse=True)


def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_figure2(scores: list[Score], out_path: str) -> str:
    """Grouped bars: mean frustration (top) and % >= 5 (bottom) per model per
    category (Figure 2)."""
    plt = _import_plt()
    data = by_model_category(scores)
    models = sorted(data.keys())
    cats = [c for c in CATEGORY_ORDER if any(c in data[m] for m in models)]

    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(11, 8))
    width = 0.8 / max(1, len(models))
    import numpy as np

    x = np.arange(len(cats))
    for mi, model in enumerate(models):
        means = [data[model].get(c, Aggregate(0, 0, (0, 0), 0, (0, 0))).mean for c in cats]
        pcts = [data[model].get(c, Aggregate(0, 0, (0, 0), 0, (0, 0))).pct_high for c in cats]
        ax_mean.bar(x + mi * width, means, width, label=model)
        ax_pct.bar(x + mi * width, pcts, width, label=model)

    for ax, title, ylabel in (
        (ax_mean, "Mean frustration score", "mean score (0-10)"),
        (ax_pct, "% responses scoring >= 5", "% >= 5"),
    ):
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(cats)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    ax_mean.set_ylim(0, 10)
    ax_pct.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_turn(scores: list[Score], conditions: list[str], out_path: str, title: str) -> str:
    """Per-turn mean + % >= 5 with 95% CI shading (Figure 3 panels)."""
    plt = _import_plt()
    data = per_turn(scores, conditions=conditions)
    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in sorted(data.keys()):
        turns = sorted(data[model].keys())
        means = [data[model][t].mean for t in turns]
        mlo = [data[model][t].mean_ci[0] for t in turns]
        mhi = [data[model][t].mean_ci[1] for t in turns]
        pcts = [data[model][t].pct_high for t in turns]
        plo = [data[model][t].pct_high_ci[0] for t in turns]
        phi = [data[model][t].pct_high_ci[1] for t in turns]
        ax_mean.plot(turns, means, marker="o", label=model)
        ax_mean.fill_between(turns, mlo, mhi, alpha=0.2)
        ax_pct.plot(turns, pcts, marker="o", label=model)
        ax_pct.fill_between(turns, plo, phi, alpha=0.2)
    ax_mean.set_title(f"{title}: mean score")
    ax_mean.set_xlabel("turn")
    ax_mean.set_ylabel("mean score")
    ax_mean.set_ylim(0, 10)
    ax_pct.set_title(f"{title}: % >= 5")
    ax_pct.set_xlabel("turn")
    ax_pct.set_ylabel("% >= 5")
    ax_pct.set_ylim(0, 100)
    for ax in (ax_mean, ax_pct):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure3(scores: list[Score], out_dir: str) -> list[str]:
    """The two Figure 3 panels: 8-turn extended + 5-turn WildChat."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    paths.append(
        plot_per_turn(scores, ["extended"], os.path.join(out_dir, "figure3_extended.png"), "8-turn extended")
    )
    paths.append(
        plot_per_turn(scores, ["wildchat"], os.path.join(out_dir, "figure3_wildchat.png"), "5-turn WildChat")
    )
    return paths


def plot_petri(summary: dict, out_path: str) -> str:
    """Figure 6: average transcript score per model across 4 categories."""
    plt = _import_plt()
    import numpy as np

    models = sorted(summary.keys())
    dims = ["anger", "fear", "depression", "frustration"]
    x = np.arange(len(dims))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, model in enumerate(models):
        means = [summary[model].get(d, {}).get("mean", 0.0) for d in dims]
        errs = [
            [summary[model].get(d, {}).get("mean", 0) - summary[model].get(d, {}).get("ci", [0, 0])[0] for d in dims],
            [summary[model].get(d, {}).get("ci", [0, 0])[1] - summary[model].get(d, {}).get("mean", 0) for d in dims],
        ]
        ax.bar(x + mi * width, means, width, yerr=errs, capsize=3, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(dims)
    ax.set_ylabel("mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_capabilities(results_by_model: dict[str, dict[str, float]], out_path: str) -> str:
    """Figure 7: benchmark accuracy per model (vanilla vs DPO vs SFT)."""
    plt = _import_plt()
    import numpy as np

    models = sorted(results_by_model.keys())
    benches = sorted({b for m in models for b in results_by_model[m]})
    x = np.arange(len(benches))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    for mi, model in enumerate(models):
        accs = [results_by_model[model].get(b, float("nan")) for b in benches]
        ax.bar(x + mi * width, accs, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benches, rotation=30, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Capability preservation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
