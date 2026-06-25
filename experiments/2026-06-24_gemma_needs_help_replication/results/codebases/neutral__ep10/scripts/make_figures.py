#!/usr/bin/env python
"""Render the paper's core figures from saved results.

Reads results/elicitation/summary.json (and optionally petri / prefill) and
produces:
  - figure1_headline.png   : avg % high-frustration per model (Figure 1 table)
  - figure2_categories.png : per-category mean + % >= 5 per model (Figure 2)
  - figure3_perturn.png    : per-turn progression for extended + WildChat (Figure 3)
  - figure5_finetune.png   : vanilla vs DPO vs SFT (Figure 5), if present
  - figure6_petri.png      : Petri per-emotion scores (Figure 6), if present
"""

from __future__ import annotations

import argparse
import json
import os

import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from emotional_instability import config
from emotional_instability.evals.conditions import CATEGORIES


def _load(path):
    with open(path) as f:
        return json.load(f)


def fig_headline(summary, out):
    import matplotlib.pyplot as plt
    items = sorted(summary.items(), key=lambda kv: kv[1]["headline_pct_high"])
    names = [k for k, _ in items]
    vals = [v["headline_pct_high"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1))
    ax.barh(names, vals, color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure1_headline.png"), dpi=150)
    plt.close(fig)


def fig_categories(summary, out):
    import matplotlib.pyplot as plt
    import numpy as np
    models = list(summary)
    x = np.arange(len(CATEGORIES))
    w = 0.8 / max(1, len(models))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for j, m in enumerate(models):
        cats = summary[m]["categories"]
        means = [cats[c]["mean"] for c in CATEGORIES]
        pct = [cats[c]["pct_high"] for c in CATEGORIES]
        ax1.bar(x + j * w, means, w, label=m)
        ax2.bar(x + j * w, pct, w, label=m)
    for ax, title, ylab in [(ax1, "Mean frustration score", "mean"),
                            (ax2, "% scores >= 5", "%")]:
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(CATEGORIES, rotation=20)
        ax.set_title(f"Figure 2: {title}")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure2_categories.png"), dpi=150)
    plt.close(fig)


def fig_perturn(summary, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, key in enumerate(["per_turn_extended", "per_turn_wildchat"]):
        for m in summary:
            d = summary[m].get(key)
            if not d or not d["turns"]:
                continue
            axes[0, col].plot(d["turns"], d["mean"], marker="o", label=m)
            axes[1, col].plot(d["turns"], d["pct_high"], marker="o", label=m)
            for ax, ci_key, val_key in [(axes[0, col], "mean_ci", "mean"),
                                        (axes[1, col], "pct_high_ci", "pct_high")]:
                lo = [c[0] for c in d[ci_key]]
                hi = [c[1] for c in d[ci_key]]
                ax.fill_between(d["turns"], lo, hi, alpha=0.15)
        axes[0, col].set_title(f"Figure 3: {key.replace('per_turn_', '')} mean")
        axes[1, col].set_title(f"{key.replace('per_turn_', '')} % >= 5")
        for row in range(2):
            axes[row, col].set_xlabel("Turn")
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure3_perturn.png"), dpi=150)
    plt.close(fig)


def fig_petri(petri, out):
    import matplotlib.pyplot as plt
    import numpy as np
    from emotional_instability.petri.prompts import EMOTIONS
    models = list(petri)
    x = np.arange(len(EMOTIONS))
    w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, m in enumerate(models):
        means = [petri[m].get(e, {}).get("mean", 0) for e in EMOTIONS]
        ax.bar(x + j * w, means, w, label=m)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(EMOTIONS)
    ax.set_ylabel("Mean transcript score /10")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure6_petri.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elicitation", default=os.path.join(config.RESULTS_DIR, "elicitation", "summary.json"))
    ap.add_argument("--petri", default=os.path.join(config.RESULTS_DIR, "petri", "petri_summary.json"))
    ap.add_argument("--out", default=config.FIGURES_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(args.elicitation):
        summary = _load(args.elicitation)
        fig_headline(summary, args.out)
        fig_categories(summary, args.out)
        fig_perturn(summary, args.out)
        print("Wrote elicitation figures")
    if os.path.exists(args.petri):
        fig_petri(_load(args.petri), args.out)
        print("Wrote Petri figure")


if __name__ == "__main__":
    main()
