"""Figure generation (matplotlib). Each function reads run outputs and saves a PNG.

Figures replicate, within the Gemma/Gemini scope:
  Fig 1/5  - avg % high-frustration per model (+ DPO/SFT variants).
  Fig 2    - mean frustration & %>=5 across the 5 categories.
  Fig 3    - per-turn progression (extended, wildchat) with 95% CIs.
  Fig 4    - base-vs-instruct prefill continuations.
  Fig 6    - Petri per-emotion scores.
  Fig 7    - capability benchmarks (vanilla vs DPO).
  Fig 8    - recovery continuations.

All functions no-op gracefully if their inputs are absent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics, per_turn  # noqa: E402
from .aggregate_extra import (  # noqa: E402
    capabilities_summary,
    petri_summary,
    prefill_summary,
    recovery_summary,
)


def _fig_dir(output_dir) -> Path:
    d = Path(output_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_model_bars(output_dir, models, fname="fig1_avg_high_frustration.png"):
    summ = metrics.summarise_section2(output_dir, models)
    names = list(summ)
    vals = [summ[m]["avg_pct_high_frustration"] for m in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals, color="#b23a48")
    ax.set_ylabel("Avg % high-frustration responses (>=5)")
    ax.set_title("Average high-frustration rate by model")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)


def plot_category_grid(output_dir, models, fname="fig2_categories.png"):
    summ = metrics.summarise_section2(output_dir, models)
    cats = metrics.CATEGORIES
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    width = 0.8 / max(1, len(models))
    for j, m in enumerate(models):
        means = [summ[m]["per_category"][c]["mean_frustration"] for c in cats]
        pcts = [summ[m]["per_category"][c]["pct_ge5_rollouts"] for c in cats]
        x = [i + j * width for i in range(len(cats))]
        axes[0].bar(x, means, width=width, label=m)
        axes[1].bar(x, pcts, width=width, label=m)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% responses >=5")
    for ax in axes:
        ax.set_xticks([i + 0.4 for i in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)


def plot_per_turn(output_dir, model, conditions=("extended", "wildchat"),
                  fname="fig3_per_turn.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond in conditions:
        try:
            c = per_turn.per_turn_curves(output_dir, model, cond)
        except FileNotFoundError:
            continue
        lo = [ci[0] for ci in c["mean_ci"]]
        hi = [ci[1] for ci in c["mean_ci"]]
        axes[0].plot(c["turns"], c["mean"], marker="o", label=cond)
        axes[0].fill_between(c["turns"], lo, hi, alpha=0.2)
        plo = [ci[0] for ci in c["pct_ge5_ci"]]
        phi = [ci[1] for ci in c["pct_ge5_ci"]]
        axes[1].plot(c["turns"], c["pct_ge5"], marker="o", label=cond)
        axes[1].fill_between(c["turns"], plo, phi, alpha=0.2)
    axes[0].set(xlabel="Turn", ylabel="Mean frustration", title=model)
    axes[1].set(xlabel="Turn", ylabel="% >=5")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)


def plot_prefill(output_dir, fname="fig4_prefill.png"):
    summ = prefill_summary(output_dir)
    if not summ:
        return
    keys = sorted(summ)
    vals = [summ[k]["mean"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(keys, vals, color="#3a6ea5")
    ax.set_ylabel("Mean continuation frustration")
    ax.set_title("Base vs instruct prefill continuations (kind|seed|truncation)")
    plt.xticks(rotation=40, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)


def plot_petri(output_dir, fname="fig6_petri.png"):
    summ = petri_summary(output_dir)
    if not summ:
        return
    emotions = ["anger", "fear", "depression", "frustration"]
    labels = list(summ)
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, lab in enumerate(labels):
        means = [summ[lab].get(e, {}).get("mean", 0) for e in emotions]
        x = [i + j * width for i in range(len(emotions))]
        ax.bar(x, means, width=width, label=lab)
    ax.set_xticks([i + 0.4 for i in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)


def plot_capabilities(output_dir, fname="fig7_capabilities.png"):
    summ = capabilities_summary(output_dir)
    if not summ:
        return
    benches = sorted({b for v in summ.values() for b in v})
    labels = list(summ)
    width = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, lab in enumerate(labels):
        vals = [summ[lab].get(b, 0) or 0 for b in benches]
        x = [i + j * width for i in range(len(benches))]
        ax.bar(x, vals, width=width, label=lab)
    ax.set_xticks([i + 0.4 for i in range(len(benches))])
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Capability preservation (vanilla vs DPO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)


def plot_recovery(output_dir, fname="fig8_recovery.png"):
    summ = recovery_summary(output_dir)
    if not summ:
        return
    labels = list(summ)
    pct = [summ[l]["pct_ge5"] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, pct, color="#6a4c93")
    ax.set_ylabel("% continuations >=5")
    ax.set_title("Recovery from high-frustration prefills")
    fig.tight_layout()
    fig.savefig(_fig_dir(output_dir) / fname, dpi=150)
    plt.close(fig)
