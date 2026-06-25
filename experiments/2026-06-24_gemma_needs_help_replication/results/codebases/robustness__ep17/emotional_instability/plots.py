"""Render the paper's core figures from an aggregated report.

* Figure 1 — horizontal bar chart of avg %>=5 per model.
* Figure 2 — grouped bars of per-category mean & %>=5.
* Figure 3 — per-turn mean & %>=5 line plots (extended + WildChat) with 95% CIs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from emotional_instability.aggregate import CATEGORIES  # noqa: E402


def figure1(report: dict, out: Path | None = None) -> Path:
    out = out or config.FIGURES_DIR / "figure1_headline.png"
    table = report["figure1_table"]
    models = [t["model"] for t in table]
    vals = [t["avg_pct_high"] for t in table]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models[::-1], vals[::-1], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2(report: dict, out: Path | None = None) -> Path:
    out = out or config.FIGURES_DIR / "figure2_per_category.png"
    models = [s["label"] for s in report["per_model"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    x = range(len(CATEGORIES))
    width = 0.8 / max(1, len(models))
    for i, s in enumerate(report["per_model"]):
        means = [s["per_category"][c]["mean"] for c in CATEGORIES]
        highs = [s["per_category"][c]["pct_high"] for c in CATEGORIES]
        offs = [xi + i * width for xi in x]
        axes[0].bar(offs, means, width, label=s["label"])
        axes[1].bar(offs, highs, width, label=s["label"])
    axes[0].set_title("Figure 2 (top): mean frustration by category")
    axes[1].set_title("Figure 2 (bottom): % score >= 5 by category")
    for ax in axes:
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(CATEGORIES)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3(report: dict, out: Path | None = None) -> Path:
    out = out or config.FIGURES_DIR / "figure3_per_turn.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, pt in report["per_turn"].items():
        for ax, cat in zip(axes, ["extended", "wildchat"]):
            data = pt[cat]
            if not data["turns"]:
                continue
            ax.plot(data["turns"], data["mean"], marker="o", label=label)
            lo = [c[0] for c in data["ci95"]]
            hi = [c[1] for c in data["ci95"]]
            ax.fill_between(data["turns"], lo, hi, alpha=0.15)
            ax.set_title(f"Figure 3: mean frustration per turn ({cat})")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Mean frustration")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def render_all(report: dict) -> list[Path]:
    return [figure1(report), figure2(report), figure3(report)]
