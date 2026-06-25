"""Aggregation and figure generation reproducing the paper's headline plots.

  Figure 1/2: avg % high-frustration and mean frustration per model.
  Figure 3:   per-turn frustration trajectories (8-turn + WildChat).
  Figure 5:   vanilla vs SFT vs DPO comparison.
  Figure 6:   Petri mean score per emotion dimension per model.

All figures read the JSONL result files written by the eval modules, so analysis
can be re-run without re-querying any model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import RESULTS_DIR
from .evaluation import aggregate, load_records


def _maybe_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def summarize_section2(results_dir: Path | None = None) -> dict:
    """Build the Figure 1/2 table: per-model avg % >= 5 and mean frustration."""
    results_dir = results_dir or (RESULTS_DIR / "section2")
    summary = {}
    for path in sorted(results_dir.glob("*_responses.jsonl")):
        recs = load_records(path)
        summary[recs[0].model] = aggregate(recs)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_model_comparison(summary: dict, out_path: Path) -> None:
    plt = _maybe_plt()
    if plt is None:
        return
    models = list(summary.keys())
    pct = [summary[m]["overall_pct_high"] for m in models]
    mean = [summary[m]["overall_mean_frustration"] for m in models]
    order = np.argsort(pct)[::-1]
    models = [models[i] for i in order]
    pct = [pct[i] for i in order]
    mean = [mean[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(models, pct, color="indianred")
    axes[0].set_ylabel("% responses scoring >= 5")
    axes[0].set_title("High-frustration rate by model")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(models, mean, color="steelblue")
    axes[1].set_ylabel("Mean frustration (0-10)")
    axes[1].set_title("Mean frustration by model")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(summary: dict, out_path: Path) -> None:
    plt = _maybe_plt()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, agg in summary.items():
        for ax, cat in zip(axes, ("extended", "wildchat")):
            turns = agg["per_turn"].get(cat, {})
            if not turns:
                continue
            xs = sorted(int(t) for t in turns)
            ys = [turns[str(t) if str(t) in turns else t]["mean_frustration"] for t in xs]
            ax.plot([x + 1 for x in xs], ys, marker="o", label=model)
            ax.set_title(f"{cat}: mean frustration per turn")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Mean frustration")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intervention_comparison(variants: dict[str, dict], out_path: Path) -> None:
    """Figure 5: vanilla vs SFT vs DPO. `variants` maps label -> aggregate dict."""
    plt = _maybe_plt()
    if plt is None:
        return
    labels = list(variants.keys())
    pct = [variants[l]["overall_pct_high"] for l in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, pct, color=["gray", "orange", "green"][: len(labels)])
    ax.set_ylabel("% responses scoring >= 5")
    ax.set_title("Intervention comparison (Section 4.2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_petri(petri_summaries: list[dict], out_path: Path) -> None:
    """Figure 6: grouped bars of mean emotion score per model."""
    plt = _maybe_plt()
    if plt is None:
        return
    from .judge import EMOTION_DIMENSIONS
    models = [s["model"] for s in petri_summaries]
    x = np.arange(len(EMOTION_DIMENSIONS))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(petri_summaries):
        vals = [s["mean_by_emotion"][e] for e in EMOTION_DIMENSIONS]
        ax.bar(x + i * width, vals, width, label=s["model"])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(EMOTION_DIMENSIONS)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
