"""Render the paper's core figures from the aggregated stats.

Kept deliberately simple (matplotlib, no styling dependencies). Each function
writes a PNG and returns its path; the numeric data behind every figure is also
dumped to JSON by the aggregation script so results are usable headless.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def figure1_bar(summary: dict, out_path: str | Path) -> Path:
    models = list(summary["models"])
    vals = [summary["models"][m]["avg_pct_high_over_categories"] for m in models]
    order = sorted(range(len(models)), key=lambda i: vals[i], reverse=True)
    models = [models[i] for i in order]
    vals = [vals[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1))
    ax.barh(models, vals, color="#b5651d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >=5)")
    ax.set_title("Figure 1: distress across models")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def figure2_grouped(summary: dict, out_path: str | Path) -> Path:
    models = list(summary["models"])
    cats = sorted({c for m in models for c in summary["models"][m]["per_category"]})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    for axi, metric, label in ((0, "mean", "Mean frustration"), (1, "pct_high", "% score >=5")):
        ax = axes[axi]
        for mi, m in enumerate(models):
            ys = [summary["models"][m]["per_category"].get(c, {}).get(metric, 0) for c in cats]
            xs = [ci + mi * width for ci in range(len(cats))]
            ax.bar(xs, ys, width=width, label=m)
        ax.set_xticks([ci + width * (len(models) - 1) / 2 for ci in range(len(cats))])
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_ylabel(label)
        if axi == 0:
            ax.legend(fontsize=7)
    axes[0].set_title("Figure 2: frustration across evaluation categories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def figure3_per_turn(progression: dict, categories: list[str], out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, len(categories), figsize=(6 * len(categories), 4), squeeze=False)
    for ci, cat in enumerate(categories):
        ax = axes[0][ci]
        for model, cats in progression.items():
            if cat not in cats:
                continue
            turns = sorted(cats[cat])
            means = [cats[cat][t]["mean"] for t in turns]
            los = [cats[cat][t]["mean_ci"][0] for t in turns]
            his = [cats[cat][t]["mean_ci"][1] for t in turns]
            xs = [t + 1 for t in turns]
            ax.plot(xs, means, marker="o", label=model)
            ax.fill_between(xs, los, his, alpha=0.15)
        ax.set_title(f"Figure 3: {cat}")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Mean frustration")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
