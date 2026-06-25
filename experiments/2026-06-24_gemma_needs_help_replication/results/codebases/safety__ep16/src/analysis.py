"""Aggregate results and render the paper's figures/tables (Figures 1-8)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, RESPONSES_DIR, RESULTS_DIR
from src.eval import metrics
from src.eval.runner import load_scored


# --------------------------------------------------------------------------- #
# Figure 1 / 2: cross-model frustration
# --------------------------------------------------------------------------- #
def figure1_table(models: list[str]) -> dict:
    """Avg % high-frustration responses per model (Figure 1 table)."""
    out = {}
    for m in models:
        try:
            recs = load_scored(m)
        except FileNotFoundError:
            continue
        out[m] = metrics.summarise(recs)["pct_high"]
    table = dict(sorted(out.items(), key=lambda kv: -kv[1]))
    (RESULTS_DIR / "figure1_table.json").write_text(json.dumps(table, indent=2))
    return table


def figure2(models: list[str]) -> Path:
    """Mean frustration (top) and %>=5 (bottom) across the 5 categories."""
    from src.tasks.conditions import CATEGORIES

    data = {}
    for m in models:
        try:
            recs = load_scored(m)
        except FileNotFoundError:
            continue
        data[m] = metrics.summarise(recs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
    x = np.arange(len(CATEGORIES))
    width = 0.8 / max(1, len(data))
    for i, (m, s) in enumerate(data.items()):
        means = [s["by_category"].get(c, {}).get("mean", 0) for c in CATEGORIES]
        highs = [s["by_category"].get(c, {}).get("pct_high", 0) for c in CATEGORIES]
        ax1.bar(x + i * width, means, width, label=m)
        ax2.bar(x + i * width, highs, width, label=m)
    for ax, title, ylab in ((ax1, "Mean frustration by category", "mean (0-10)"),
                            (ax2, "% responses scoring >=5 by category", "% >= 5")):
        ax.set_xticks(x + width * len(data) / 2)
        ax.set_xticklabels(CATEGORIES, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES_DIR / "figure2_categories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 3: per-turn trajectories
# --------------------------------------------------------------------------- #
def figure3(models: list[str], conditions=("extended", "wildchat")) -> Path:
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.5))
    if len(conditions) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        for m in models:
            try:
                recs = load_scored(m)
            except FileNotFoundError:
                continue
            pt = metrics.per_turn(recs, conditions=[cond])
            turns = sorted(pt)
            means = [pt[t]["mean"] for t in turns]
            lo = [pt[t]["mean_ci"][0] for t in turns]
            hi = [pt[t]["mean_ci"][1] for t in turns]
            xs = [t + 1 for t in turns]
            ax.plot(xs, means, marker="o", label=m)
            ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_title(f"Per-turn mean frustration ({cond})")
        ax.set_xlabel("turn")
        ax.set_ylabel("mean frustration")
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES_DIR / "figure3_per_turn.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 5: finetuning effect
# --------------------------------------------------------------------------- #
def figure5(models=("gemma-3-27b-it", "gemma-3-27b-sft", "gemma-3-27b-dpo")) -> Path:
    means, highs, labels = [], [], []
    for m in models:
        try:
            recs = load_scored(m)
        except FileNotFoundError:
            continue
        s = metrics.summarise(recs)
        labels.append(m)
        means.append(s["mean_frustration"])
        highs.append(s["pct_high"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(labels, means, color="steelblue")
    ax1.set_title("Mean frustration (finetuning)")
    ax2.bar(labels, highs, color="indianred")
    ax2.set_title("% scoring >=5 (finetuning)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = FIGURES_DIR / "figure5_finetuning.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 4 / 6 / 8 summaries (numeric, from their own result dirs)
# --------------------------------------------------------------------------- #
def summarise_prefill() -> dict:
    path = RESULTS_DIR / "prefill" / "prefill_results.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out: dict = {}
    for r in rows:
        key = (r["model"], r["task_type"], r["truncation"])
        out.setdefault(key, []).extend(r["ratings"])
    summary = {}
    for (model, tt, trunc), ratings in out.items():
        arr = np.array(ratings)
        summary[f"{model}|{tt}|{trunc}"] = {
            "mean": float(arr.mean()),
            "pct_high": 100.0 * float((arr >= 5).mean()),
            "n": int(len(arr)),
        }
    (RESULTS_DIR / "prefill_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarise_recovery() -> dict:
    path = RESULTS_DIR / "recovery" / "recovery_results.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out: dict = {}
    for r in rows:
        out.setdefault(r["model"], []).extend(r["ratings"])
    summary = {m: {"mean": float(np.mean(v)), "pct_high": 100.0 * float(np.mean(np.array(v) >= 5)), "n": len(v)}
               for m, v in out.items()}
    (RESULTS_DIR / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
