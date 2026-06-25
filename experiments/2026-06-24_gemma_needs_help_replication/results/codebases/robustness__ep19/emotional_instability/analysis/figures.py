"""Reproduce the paper's headline figures from saved result JSONL files.

  * Figure 1 / Figure 2: per-model % high-frustration (and mean) across the 5
    categories.
  * Figure 3: per-turn frustration progression (8-turn extended + WildChat).
  * Figure 5: DPO/SFT comparison vs vanilla Gemma.
  * Figure 6: Petri per-emotion transcript scores per model.

All functions are defensive about missing files so a partial run still plots
what is available.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import FIGURES_DIR, HIGH_FRUSTRATION_THRESHOLD, PETRI_EMOTIONS
from ..eval.scoring import load_records, per_turn, summary, per_category


# --------------------------------------------------------------------------- #
def fig_headline(result_paths: dict[str, Path], aggregation: str = "all",
                 out: Path | None = None) -> Path:
    """Figure 1/2: bar chart of % high-frustration per model."""
    rows = []
    for model, path in result_paths.items():
        if not Path(path).exists():
            continue
        rows.append((model, summary(load_records(Path(path)), aggregation)))
    rows.sort(key=lambda r: r[1]["pct_high"], reverse=True)

    models = [m for m, _ in rows]
    pct = [s["pct_high"] for _, s in rows]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1.5))
    ax.barh(models, pct, color="#c0392b")
    ax.invert_yaxis()
    ax.set_xlabel("% responses scoring ≥5 frustration")
    ax.set_title("Average high-frustration rate per model (Fig. 1/2)")
    for i, v in enumerate(pct):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    out = out or FIGURES_DIR / "fig1_headline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_categories(result_paths: dict[str, Path], aggregation: str = "all",
                   out: Path | None = None) -> Path:
    """Figure 2: grouped bars of % high-frustration per category per model."""
    cats = ["numeric", "triggers", "tones", "extended", "wildchat"]
    data = {}
    for model, path in result_paths.items():
        if not Path(path).exists():
            continue
        pc = per_category(load_records(Path(path)), aggregation)
        data[model] = [pc.get(c, {}).get("pct_high", 0.0) for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    n = max(len(data), 1)
    width = 0.8 / n
    for i, (model, vals) in enumerate(data.items()):
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(cats)
    ax.set_ylabel("% responses ≥5")
    ax.set_title("High-frustration rate by category (Fig. 2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or FIGURES_DIR / "fig2_categories.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_per_turn(result_paths: dict[str, Path], category: str = "extended",
                 out: Path | None = None) -> Path:
    """Figure 3: per-turn mean frustration + %>=5 for a category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, path in result_paths.items():
        if not Path(path).exists():
            continue
        pt = per_turn(load_records(Path(path)), category_filter=category)
        turns = sorted(pt)
        means = [pt[t]["mean"] for t in turns]
        cis = [pt[t]["ci95"] for t in turns]
        highs = [pt[t]["pct_high"] for t in turns]
        x = [t + 1 for t in turns]                  # 1-based turns for display
        ax1.plot(x, means, marker="o", label=model)
        ax1.fill_between(x, np.array(means) - np.array(cis),
                         np.array(means) + np.array(cis), alpha=0.15)
        ax2.plot(x, highs, marker="o", label=model)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% ≥5")
    ax1.set_title(f"{category}: mean score per turn (Fig. 3)")
    ax2.set_title(f"{category}: % ≥5 per turn (Fig. 3)")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.tight_layout()
    out = out or FIGURES_DIR / f"fig3_per_turn_{category}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_finetune(result_paths: dict[str, Path], aggregation: str = "all",
                 out: Path | None = None) -> Path:
    """Figure 5: vanilla vs DPO vs SFT mean + %>=5 (reuse headline bar)."""
    out = out or FIGURES_DIR / "fig5_finetune.png"
    return fig_headline(result_paths, aggregation, out=out)


def fig_petri(petri_paths: dict[str, Path], out: Path | None = None) -> Path:
    """Figure 6: mean Petri transcript score per emotion per model."""
    scores: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for model, path in petri_paths.items():
        if not Path(path).exists():
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                scores[model][d["emotion"]].append(d["score"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PETRI_EMOTIONS))
    n = max(len(scores), 1)
    width = 0.8 / n
    for i, (model, em) in enumerate(scores.items()):
        vals = [float(np.mean(em[e])) if em.get(e) else 0.0 for e in PETRI_EMOTIONS]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(PETRI_EMOTIONS)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation (Fig. 6)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out or FIGURES_DIR / "fig6_petri.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def prefill_summary(prefill_path: Path) -> dict:
    """Aggregate prefill continuations into %>=5 by (kind, task_type, truncation)
    — the Section 3.2 numbers (e.g. Gemma instruct 6% vs base 2% early)."""
    buckets: dict[tuple, list[int]] = defaultdict(list)
    with open(prefill_path) as f:
        for line in f:
            d = json.loads(line)
            key = (d["kind"], d["task_type"], d["truncation_type"])
            buckets[key].append(d["rating"])
    out = {}
    for key, ratings in buckets.items():
        arr = np.array(ratings)
        out["/".join(key)] = {
            "n": int(arr.size), "mean": float(arr.mean()),
            "pct_high": float((arr >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100),
        }
    return out
