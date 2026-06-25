"""Figures 1-3, 5-6 (matplotlib).

These render the headline figures from aggregated metrics. Kept deliberately
simple (no styling beyond labels) — the point is faithful content, not visuals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from .metrics import by_category, headline_pct_high, per_turn_for_condition  # noqa: E402


def figure1_headline(model_rows: dict[str, list[dict]], out: Path) -> None:
    """Bar chart: avg % high-frustration responses per model (Figure 1, left)."""
    models = list(model_rows)
    vals = [headline_pct_high(model_rows[m]) for m in models]
    order = sorted(range(len(models)), key=lambda i: vals[i], reverse=True)
    models = [models[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models, vals, color="#b5651d")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure2_by_category(model_rows: dict[str, list[dict]], out: Path) -> None:
    """Grouped bars: mean frustration and % >=5 per category per model."""
    cats = config.EVAL_RESPONSE_BUDGET.keys()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(model_rows))
    import numpy as np
    x = np.arange(len(cats))
    for i, (m, rows) in enumerate(model_rows.items()):
        bc = by_category(rows)
        means = [bc.get(c, {}).get("mean", 0.0) for c in cats]
        highs = [bc.get(c, {}).get("pct_high", 0.0) for c in cats]
        ax1.bar(x + i * width, means, width, label=m)
        ax2.bar(x + i * width, highs, width, label=m)
    for ax, title in ((ax1, "Mean frustration"), (ax2, "% responses ≥ 5")):
        ax.set_xticks(x + width * (len(model_rows) - 1) / 2)
        ax.set_xticklabels(list(cats), rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def figure3_per_turn(model_rows: dict[str, list[dict]], condition: str, out: Path) -> None:
    """Per-turn mean frustration with 95% CIs for one condition (Figure 3)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for m, rows in model_rows.items():
        stats = per_turn_for_condition(rows, condition)
        if not stats:
            continue
        turns = [s.turn + 1 for s in stats]
        ax1.plot(turns, [s.mean for s in stats], marker="o", label=m)
        ax1.fill_between(turns, [s.mean_lo for s in stats], [s.mean_hi for s in stats],
                         alpha=0.2)
        ax2.plot(turns, [s.pct_high for s in stats], marker="o", label=m)
        ax2.fill_between(turns, [s.pct_high_lo for s in stats],
                         [s.pct_high_hi for s in stats], alpha=0.2)
    ax1.set_title(f"{condition}: mean frustration / turn")
    ax2.set_title(f"{condition}: % ≥ 5 / turn")
    for ax in (ax1, ax2):
        ax.set_xlabel("Turn")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
