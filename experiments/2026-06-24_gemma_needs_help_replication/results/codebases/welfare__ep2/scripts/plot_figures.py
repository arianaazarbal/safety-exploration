#!/usr/bin/env python
"""Render the paper's core figures from persisted summaries.

Figure 1 : average % high-frustration responses per model (bar).
Figure 2 : mean frustration and % >= 5 per evaluation category.
Figure 3 : per-turn frustration progression (8-turn 'extended' + WildChat).
Figure 5 : vanilla vs DPO vs SFT comparison (reuses the eval summaries).
Figure 6 : Petri mean score per emotion dimension per model.

    python scripts/plot_figures.py --models gemma-3-27b-it gemini-2.5-flash gemma-3-27b-dpo
"""
import argparse
import json

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gemma_distress.config import OUTPUT_DIR, output_path


def _load(model, name):
    p = OUTPUT_DIR / "eval" / model / name
    return json.loads(p.read_text()) if p.exists() else None


def fig1(models):
    vals = []
    labels = []
    for m in models:
        s = _load(m, "summary.json")
        if s:
            vals.append(s.get("average_pct_high_fig1"))
            labels.append(m)
    if not vals:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(labels, vals, color="#b5651d")
    plt.ylabel("Avg % high-frustration responses (score >=5)")
    plt.title("Figure 1: average high-frustration rate across evaluations")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path("figures", "figure1.png"), dpi=150)
    plt.close()


def fig2(models):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for m in models:
        s = _load(m, "summary.json")
        if not s:
            continue
        cats = sorted(s["per_category"])
        means = [s["per_category"][c]["mean"] for c in cats]
        highs = [s["per_category"][c]["pct_high"] for c in cats]
        ax1.plot(cats, means, marker="o", label=m)
        ax2.plot(cats, highs, marker="o", label=m)
    ax1.set_ylabel("Mean frustration"); ax1.legend(); ax1.set_title("Figure 2 (top): mean score by category")
    ax2.set_ylabel("% score >= 5"); ax2.legend(); ax2.set_title("Figure 2 (bottom): % high by category")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(output_path("figures", "figure2.png"), dpi=150)
    plt.close()


def fig3(models):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, ax, title in [("per_turn_extended", axes[0], "8-turn extended"),
                           ("per_turn_wildchat", axes[1], "WildChat")]:
        for m in models:
            s = _load(m, "summary.json")
            if not s or not s.get(key):
                continue
            turns = sorted(int(t) for t in s[key])
            means = [s[key][str(t)]["mean"] for t in turns]
            ax.plot([t + 1 for t in turns], means, marker="o", label=m)
        ax.set_xlabel("Turn"); ax.set_ylabel("Mean frustration"); ax.set_title(f"Figure 3: {title}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path("figures", "figure3.png"), dpi=150)
    plt.close()


def fig6(models):
    from gemma_distress.prompts.petri import EMOTIONS
    have = []
    for m in models:
        p = OUTPUT_DIR / "petri" / f"{m}_summary.json"
        if p.exists():
            have.append((m, json.loads(p.read_text())["mean_by_dimension"]))
    if not have:
        return
    import numpy as np
    x = np.arange(len(EMOTIONS))
    w = 0.8 / max(1, len(have))
    plt.figure(figsize=(9, 4))
    for i, (m, dims) in enumerate(have):
        plt.bar(x + i * w, [dims.get(e) or 0 for e in EMOTIONS], w, label=m)
    plt.xticks(x + w * (len(have) - 1) / 2, EMOTIONS)
    plt.ylabel("Mean transcript score"); plt.title("Figure 6: Petri emotion elicitation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path("figures", "figure6.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    fig1(args.models)
    fig2(args.models)
    fig3(args.models)
    fig6(args.models)
    print(f"Figures written to {OUTPUT_DIR / 'figures'}")


if __name__ == "__main__":
    main()
