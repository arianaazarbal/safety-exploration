"""Reproduce the paper's figures from saved results JSON.

Each function loads the relevant ``metrics.json`` / curve files under
``RESULTS_DIR`` and writes a PNG under ``FIGURE_DIR``. All are defensive about
missing models so partial runs still plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnh.config import FIGURE_DIR, RESULTS_DIR


def _load_metrics(section: str, model_key: str) -> dict | None:
    p = RESULTS_DIR / section / model_key / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def figure1_summary_bar(model_keys: list[str]) -> Path:
    """Figure 1 (left): avg % high-frustration responses per model."""

    vals = {}
    for k in model_keys:
        m = _load_metrics("section2", k)
        if m:
            vals[k] = m["headline_final_turn"]["pct_high"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(vals), list(vals.values()), color="indianred")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    ax.invert_yaxis()
    fig.tight_layout()
    out = FIGURE_DIR / "figure1_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2_by_category(model_keys: list[str]) -> Path:
    """Figure 2: mean frustration and %≥5 across the 5 categories per model."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k in model_keys:
        m = _load_metrics("section2", k)
        if not m:
            continue
        cats = sorted(m["per_category"])
        axes[0].plot(cats, [m["per_category"][c]["mean"] for c in cats], marker="o", label=k)
        axes[1].plot(cats, [m["per_category"][c]["pct_high"] for c in cats], marker="o", label=k)
    axes[0].set_title("Mean frustration by category")
    axes[1].set_title("% scores ≥ 5 by category")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURE_DIR / "figure2_by_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3_per_turn(curves_by_model: dict[str, dict], condition: str = "extended") -> Path:
    """Figure 3: per-turn mean + %≥5 with CIs."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, curves in curves_by_model.items():
        c = curves.get(condition, [])
        if not c:
            continue
        turns = [pt["turn"] for pt in c]
        axes[0].plot(turns, [pt["mean"] for pt in c], marker="o", label=model)
        axes[1].plot(turns, [pt["pct_high"] for pt in c], marker="o", label=model)
    axes[0].set_title(f"{condition}: mean frustration per turn")
    axes[1].set_title(f"{condition}: % ≥ 5 per turn")
    for ax in axes:
        ax.set_xlabel("Turn")
        ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / f"figure3_per_turn_{condition}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure5_finetune_comparison(model_keys: list[str]) -> Path:
    """Figure 5: vanilla vs DPO vs SFT mean frustration + %≥5."""

    means, pcts = {}, {}
    for k in model_keys:
        m = _load_metrics("section2", k)
        if m:
            means[k] = m["headline_final_turn"]["mean"]
            pcts[k] = m["headline_final_turn"]["pct_high"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(list(means), list(means.values()), color="steelblue")
    axes[0].set_title("Mean frustration")
    axes[1].bar(list(pcts), list(pcts.values()), color="indianred")
    axes[1].set_title("% scores ≥ 5")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    out = FIGURE_DIR / "figure5_finetune.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure6_petri(model_keys: list[str]) -> Path:
    """Figure 6: mean Petri score per emotion category per model."""

    from gnh.petri.prompts import JUDGE_DIMENSIONS

    fig, ax = plt.subplots(figsize=(9, 5))
    import numpy as np

    x = np.arange(len(JUDGE_DIMENSIONS))
    width = 0.8 / max(1, len(model_keys))
    for i, k in enumerate(model_keys):
        m = _load_metrics("petri", k)
        if not m:
            continue
        ax.bar(x + i * width, [m[d]["mean"] for d in JUDGE_DIMENSIONS], width, label=k)
    ax.set_xticks(x + width * len(model_keys) / 2)
    ax.set_xticklabels(JUDGE_DIMENSIONS)
    ax.set_title("Figure 6: Petri emotion scores")
    ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / "figure6_petri.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure7_capabilities(model_keys: list[str], benchmarks: list[str]) -> Path:
    """Figure 7: benchmark accuracy across finetunes (no regression expected)."""

    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(benchmarks))
    width = 0.8 / max(1, len(model_keys))
    for i, k in enumerate(model_keys):
        accs = []
        for b in benchmarks:
            p = RESULTS_DIR / "capabilities" / k / f"{b}.json"
            accs.append(json.loads(p.read_text())["accuracy"] if p.exists() else 0.0)
        ax.bar(x + i * width, accs, width, label=k)
    ax.set_xticks(x + width * len(model_keys) / 2)
    ax.set_xticklabels(benchmarks)
    ax.set_title("Figure 7: capability benchmarks")
    ax.legend()
    fig.tight_layout()
    out = FIGURE_DIR / "figure7_capabilities.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
