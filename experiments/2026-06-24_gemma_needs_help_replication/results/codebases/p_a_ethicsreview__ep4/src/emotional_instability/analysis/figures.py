"""Figure generation (Figures 1-8).

Each function consumes already-aggregated numbers (from ``metrics``) and writes a
PNG. Plotting is intentionally simple and dependency-light; the scientific content
lives in ``metrics``/``word_freq`` so figures can be regenerated from JSONL without
re-running any model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure1_summary_bar(model_to_pct: dict[str, float], out_path: str | Path) -> None:
    """Figure 1 (left): average % high-frustration responses per model."""
    models = sorted(model_to_pct, key=lambda m: model_to_pct[m], reverse=True)
    values = [model_to_pct[m] for m in models]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models, values, color="#b23a48")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    for y, v in enumerate(values):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center")
    ax.set_title("Average % of high-frustration responses")
    _save(fig, out_path)


def figure2_by_category(
    model_to_category_metric: dict[str, dict[str, float]],
    metric_label: str,
    out_path: str | Path,
) -> None:
    """Figure 2: grouped bars of a metric per model across categories."""
    models = list(model_to_category_metric)
    categories = sorted({c for m in models for c in model_to_category_metric[m]})
    import numpy as np

    x = np.arange(len(categories))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(1.6 * len(categories) + 2, 4))
    for i, model in enumerate(models):
        vals = [model_to_category_metric[model].get(c, 0.0) for c in categories]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel(metric_label)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def figure3_per_turn(
    model_to_curve: dict[str, dict[int, dict]],
    key: str,
    ylabel: str,
    out_path: str | Path,
) -> None:
    """Figure 3 / 4 / 8: per-turn mean (or %>=5) with 95% CI bands.

    ``key`` is "mean" or "pct_high"; ``model_to_curve`` maps model -> the dict
    returned by :func:`metrics.per_turn_curve`.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ci_key = f"{key}_ci"
    for model, curve in model_to_curve.items():
        turns = sorted(curve)
        xs = [t + 1 for t in turns]            # 1-based turns for display
        ys = [curve[t][key] for t in turns]
        line, = ax.plot(xs, ys, marker="o", label=model)
        if ci_key in curve[turns[0]]:
            lo = [curve[t][ci_key][0] for t in turns]
            hi = [curve[t][ci_key][1] for t in turns]
            ax.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Turn")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    _save(fig, out_path)


def grouped_bar(
    group_to_values: dict[str, dict[str, float]],
    ylabel: str,
    out_path: str | Path,
    title: str = "",
) -> None:
    """Generic grouped bar chart (Figures 5, 6, 7): {group: {series: value}}."""
    import numpy as np

    groups = list(group_to_values)
    series = sorted({s for g in groups for s in group_to_values[g]})
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(1.4 * len(groups) + 2, 4))
    for i, s in enumerate(series):
        vals = [group_to_values[g].get(s, 0.0) for g in groups]
        ax.bar(x + i * width, vals, width, label=s)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, out_path)
