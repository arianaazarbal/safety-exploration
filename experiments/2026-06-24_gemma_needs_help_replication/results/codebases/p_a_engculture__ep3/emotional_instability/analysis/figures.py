"""Reproduce the paper's key figures from aggregated summaries.

  * fig1_headline     — Figure 1 left: avg % high-frustration per model (bar).
  * fig2_categories   — Figure 2: mean score + % >=5 across the 5 categories.
  * fig3_trajectories — Figure 3: per-turn mean + % >=5 with 95% CIs (8-turn, WildChat).
  * fig5_finetunes    — Figure 5: vanilla vs SFT vs DPO.
  * fig8_prefill      — Figure 8: continuation scores across prefill conditions.

All read the JSON summaries written by :mod:`.aggregate` and save PNGs under
``outputs/figures/``. Matplotlib only; no seaborn dependency.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _load_summary(config) -> dict:
    return json.loads(config.output_path("eval", "summary.json").read_text())


def fig1_headline(config, save: str | None = None) -> None:
    summary = _load_summary(config)
    models = sorted(summary, key=lambda m: -summary[m]["headline_pct_high"])
    vals = [summary[m]["headline_pct_high"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models, vals, color="#c0392b")
    ax.set_xlabel("Average % high-frustration responses (score >=5)")
    ax.invert_yaxis()
    for y, v in enumerate(vals):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(save or config.output_path("figures", "fig1_headline.png"), dpi=150)
    plt.close(fig)


def fig2_categories(config, save: str | None = None) -> None:
    summary = _load_summary(config)
    models = list(summary)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(len(models), 1)
    x = range(len(CATEGORIES))
    for i, m in enumerate(models):
        pc = summary[m]["per_category"]
        means = [pc.get(c, {}).get("mean", 0) for c in CATEGORIES]
        highs = [pc.get(c, {}).get("pct_high", 0) for c in CATEGORIES]
        off = [xi + i * width for xi in x]
        ax1.bar(off, means, width, label=m)
        ax2.bar(off, highs, width, label=m)
    ax1.set_ylabel("Mean frustration score")
    ax2.set_ylabel("% scores >= 5")
    ax2.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax2.set_xticklabels(CATEGORIES, rotation=20)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save or config.output_path("figures", "fig2_categories.png"), dpi=150)
    plt.close(fig)


def fig3_trajectories(config, categories=("extended", "wildchat"), save: str | None = None) -> None:
    summary = _load_summary(config)
    fig, axes = plt.subplots(len(categories), 2, figsize=(11, 4 * len(categories)), squeeze=False)
    for row, cat in enumerate(categories):
        for m, s in summary.items():
            traj = s["trajectories"].get(cat)
            if not traj:
                continue
            turns = [t + 1 for t in traj["turn"]]
            for col, (key, ci_key, ylabel) in enumerate(
                [("mean", "mean_ci", "Mean score"), ("pct_high", "pct_ci", "% >= 5")]
            ):
                ax = axes[row][col]
                ax.plot(turns, traj[key], marker="o", label=m)
                lo = [c[0] for c in traj[ci_key]]
                hi = [c[1] for c in traj[ci_key]]
                ax.fill_between(turns, lo, hi, alpha=0.15)
                ax.set_title(f"{cat}: {ylabel}")
                ax.set_xlabel("Turn")
        axes[row][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save or config.output_path("figures", "fig3_trajectories.png"), dpi=150)
    plt.close(fig)


def fig5_finetunes(config, model_names: list[str], save: str | None = None) -> None:
    """Compare vanilla / SFT / DPO headline metrics (expects their summaries merged)."""
    summary = _load_summary(config)
    models = [m for m in model_names if m in summary]
    vals = [summary[m]["headline_pct_high"] for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, vals, color=["#34495e", "#e67e22", "#27ae60"][: len(models)])
    ax.set_ylabel("Avg % high-frustration (score >= 5)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(save or config.output_path("figures", "fig5_finetunes.png"), dpi=150)
    plt.close(fig)


def fig8_prefill(config, models: list[str], modes=("early", "onset", "recovery"),
                 save: str | None = None) -> None:
    """% of continuations scoring >=5, per model per prefill mode."""
    import numpy as np

    from ..utils.io import load_jsonl

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(len(models), 1)
    x = range(len(modes))
    for i, m in enumerate(models):
        pct = []
        for mode in modes:
            rows = []
            for fname in (f"{m}.standard.jsonl", f"{m}.recovery.jsonl"):
                rows += [r for r in load_jsonl(config.output_path("prefill", fname))
                         if r.get("mode") == mode and r.get("rating") is not None]
            arr = np.array([r["rating"] for r in rows], float)
            pct.append(float((arr >= 5).mean() * 100) if len(arr) else 0.0)
        ax.bar([xi + i * width for xi in x], pct, width, label=m)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(modes)
    ax.set_ylabel("% continuations >= 5")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save or config.output_path("figures", "fig8_prefill.png"), dpi=150)
    plt.close(fig)
