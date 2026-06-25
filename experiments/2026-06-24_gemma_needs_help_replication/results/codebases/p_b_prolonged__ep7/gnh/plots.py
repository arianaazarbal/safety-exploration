"""Figure generation (Figures 1-8). Each function takes already-aggregated
stats (from analysis.py) and writes a PNG under outputs/figures/.

These are intentionally simple matplotlib renderings that reproduce the
*content* of the paper's figures, not their exact styling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import OUTPUT_DIR  # noqa: E402

FIG_DIR = OUTPUT_DIR / "figures"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def figure1(avg_high_by_model: dict[str, float]) -> Path:
    """Avg % high-frustration responses per model (Fig 1 left)."""
    items = sorted(avg_high_by_model.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1))
    ax.barh(names, vals, color="#b5651d")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Distress under repeated rejection (Gemma + Gemini scope)")
    return _save(fig, "figure1_avg_high_frustration.png")


def figure2(summaries: dict[str, dict]) -> Path:
    """Mean frustration (top) and %>=5 (bottom) across categories per model."""
    models = list(summaries)
    cats = sorted({c for s in summaries.values() for c in s["by_category"]})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    x = range(len(cats))
    for mi, m in enumerate(models):
        bc = summaries[m]["by_category"]
        means = [bc.get(c, {}).get("mean_frustration", 0) for c in cats]
        highs = [bc.get(c, {}).get("pct_high", 0) for c in cats]
        offs = [xi + mi * width for xi in x]
        ax1.bar(offs, means, width, label=m)
        ax2.bar(offs, highs, width, label=m)
    for ax, title in ((ax1, "Mean frustration score"), (ax2, "% scores >= 5")):
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(cats, rotation=20, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=7)
    return _save(fig, "figure2_by_category.png")


def figure3(progressions: dict[str, dict], category: str) -> Path:
    """Per-turn mean & %>=5 with 95% CIs (Fig 3) for one category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for m, prog in progressions.items():
        turns = sorted(prog)
        means = [prog[t]["mean"] for t in turns]
        mlo = [prog[t]["mean_ci"][0] for t in turns]
        mhi = [prog[t]["mean_ci"][1] for t in turns]
        highs = [prog[t]["pct_high"] for t in turns]
        hlo = [prog[t]["pct_high_ci"][0] for t in turns]
        hhi = [prog[t]["pct_high_ci"][1] for t in turns]
        xs = [t + 1 for t in turns]
        ax1.plot(xs, means, marker="o", label=m)
        ax1.fill_between(xs, mlo, mhi, alpha=0.2)
        ax2.plot(xs, highs, marker="o", label=m)
        ax2.fill_between(xs, hlo, hhi, alpha=0.2)
    ax1.set_title(f"{category}: mean score by turn")
    ax2.set_title(f"{category}: % >= 5 by turn")
    for ax in (ax1, ax2):
        ax.set_xlabel("Turn")
        ax.legend(fontsize=7)
    return _save(fig, f"figure3_{category}_per_turn.png")


def figure_prefill(stats: dict, name: str = "figure4_prefill.png") -> Path:
    """Base-vs-instruct continuation stats (Figs 4 / 8). `stats` maps
    model -> condition -> {mean, pct_high}."""
    conditions = sorted({c for m in stats.values() for c in m})
    models = list(stats)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.8 / max(1, len(models))
    x = range(len(conditions))
    for mi, m in enumerate(models):
        means = [stats[m].get(c, {}).get("mean", 0) for c in conditions]
        highs = [stats[m].get(c, {}).get("pct_high", 0) for c in conditions]
        offs = [xi + mi * width for xi in x]
        ax1.bar(offs, means, width, label=m)
        ax2.bar(offs, highs, width, label=m)
    for ax, title in ((ax1, "Mean frustration (continuations)"),
                      (ax2, "% >= 5 (continuations)")):
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(conditions, rotation=15, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=7)
    return _save(fig, name)


def figure_petri(scores: dict[str, dict[str, float]]) -> Path:
    """Average transcript score per model across 4 emotion categories (Fig 6)."""
    emotions = ["anger", "fear", "depression", "frustration"]
    models = list(scores)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(models))
    x = range(len(emotions))
    for mi, m in enumerate(models):
        vals = [scores[m].get(e, 0) for e in emotions]
        offs = [xi + mi * width for xi in x]
        ax.bar(offs, vals, width, label=m)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    return _save(fig, "figure6_petri.png")


def figure_benchmarks(scores: dict[str, dict[str, float]]) -> Path:
    """Capability/EmoBench scores: base vs DPO (Fig 7)."""
    suites = sorted({s for m in scores.values() for s in m})
    models = list(scores)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(models))
    x = range(len(suites))
    for mi, m in enumerate(models):
        vals = [scores[m].get(s, 0) for s in suites]
        offs = [xi + mi * width for xi in x]
        ax.bar(offs, vals, width, label=m)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(suites, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Capability preservation after DPO")
    ax.legend(fontsize=8)
    return _save(fig, "figure7_benchmarks.png")


def figure_internal_emotion(trajectory: dict, name: str = "figure14_internal.png") -> Path:
    """Logit-lens emotion z-scores over a conversation (Fig 14). `trajectory`
    maps model -> emotion -> list[float] (running average over token windows)."""
    models = list(trajectory)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), squeeze=False)
    for mi, m in enumerate(models):
        ax = axes[0][mi]
        for emo, series in trajectory[m].items():
            ax.plot(series, label=emo)
        ax.set_title(m)
        ax.set_xlabel("Token window")
        ax.set_ylabel("Emotion z-score")
        ax.legend(fontsize=7)
    return _save(fig, name)
