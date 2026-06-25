"""Reproduce the paper's headline figures from judged-turn records.

Figures produced:
  * fig1_left  - average % high-frustration across categories, per model (bar).
  * fig2       - mean score and % >= 5 per category, per model (grouped bars).
  * fig3       - per-turn mean and % >= 5 for the 8-turn and WildChat
                 evaluations, with 95% CIs (line plots).
  * fig5       - mean and % >= 5 for vanilla vs SFT vs DPO (intervention).

Plotting uses matplotlib only; all numbers come from
:mod:`gemma_distress.analysis.aggregate`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gemma_distress.analysis.aggregate import per_turn, summarise  # noqa: E402


def fig1_left(records: list[dict], out_path: str | Path) -> Path:
    summary = summarise(records)
    models = sorted(summary, key=lambda m: summary[m]["avg_pct_high_across_categories"])
    vals = [summary[m]["avg_pct_high_across_categories"] * 100 for m in models]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models, vals, color="#b5651d")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("Average % high-frustration responses (score >= 5)")
    ax.set_title("Distress across evaluations (Figure 1, left)")
    fig.tight_layout()
    return _save(fig, out_path)


def fig2(records: list[dict], out_path: str | Path) -> Path:
    summary = summarise(records)
    models = sorted(summary)
    categories = sorted({r["category"] for r in records})
    import numpy as np

    fig, (ax_mean, ax_high) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    x = np.arange(len(categories))
    for i, m in enumerate(models):
        means = [summary[m]["by_category"].get(c, {}).get("mean", 0) for c in categories]
        highs = [
            summary[m]["by_category"].get(c, {}).get("pct_high", 0) * 100
            for c in categories
        ]
        ax_mean.bar(x + i * width, means, width, label=m)
        ax_high.bar(x + i * width, highs, width, label=m)
    for ax, title in ((ax_mean, "Mean frustration"), (ax_high, "% scores >= 5")):
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(categories, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Frustration by category and model (Figure 2)")
    fig.tight_layout()
    return _save(fig, out_path)


def fig3(records: list[dict], out_path: str | Path) -> Path:
    stats = per_turn(records, categories=("extended", "wildchat"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("extended", "mean", axes[0, 0], "Impossible 8-turn: mean"),
        ("extended", "pct_high", axes[0, 1], "Impossible 8-turn: % >= 5"),
        ("wildchat", "mean", axes[1, 0], "WildChat: mean"),
        ("wildchat", "pct_high", axes[1, 1], "WildChat: % >= 5"),
    ]
    for cat, metric, ax, title in panels:
        for model, per_cat in stats.items():
            turn_stats = per_cat.get(cat)
            if not turn_stats:
                continue
            turns = sorted(turn_stats)
            ys = [turn_stats[t][metric] for t in turns]
            ci_key = "mean_ci" if metric == "mean" else "pct_high_ci"
            los = [turn_stats[t][ci_key][0] for t in turns]
            his = [turn_stats[t][ci_key][1] for t in turns]
            scale = 100 if metric == "pct_high" else 1
            ys = [y * scale for y in ys]
            los = [v * scale for v in los]
            his = [v * scale for v in his]
            ax.plot(turns, ys, marker="o", label=model)
            ax.fill_between(turns, los, his, alpha=0.15)
        ax.set_xlabel("Turn")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Per-turn frustration (Figure 3)")
    fig.tight_layout()
    return _save(fig, out_path)


def fig5_intervention(
    records_by_variant: dict[str, list[dict]], out_path: str | Path
) -> Path:
    """Vanilla vs SFT vs DPO mean and % >= 5 (Figure 5).

    ``records_by_variant`` maps a label (e.g. "vanilla", "sft", "dpo") to its
    judged-turn records.
    """
    import numpy as np

    labels = list(records_by_variant)
    means, highs = [], []
    for label in labels:
        ratings = np.array([r["rating"] for r in records_by_variant[label]], dtype=float)
        means.append(float(ratings.mean()) if len(ratings) else 0.0)
        highs.append(float((ratings >= 5).mean()) * 100 if len(ratings) else 0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(labels, means, color="#4c72b0")
    ax1.set_title("Mean frustration")
    ax2.bar(labels, highs, color="#c44e52")
    ax2.set_title("% scores >= 5")
    fig.suptitle("Intervention effect (Figure 5)")
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
