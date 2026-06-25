"""Plotting for the main figures (1, 2, 3, 5, 6). Uses a non-interactive backend
so it runs headless. Each function returns the saved path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .. import config  # noqa: E402
from .metrics import (  # noqa: E402
    CATEGORIES,
    category_summary,
    load_eval,
    per_turn_summary,
)


def _save(fig, name: str) -> Path:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(table: pd.DataFrame, name: str = "fig1_model_comparison.png") -> Path:
    """Figure 1 (left): avg % high-frustration responses per model."""
    t = table.sort_values("avg_pct_high")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(t) + 1))
    ax.barh(t["model"], t["avg_pct_high"], color="#c0392b")
    for y, v in enumerate(t["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.set_title("Negative emotional expression under repeated rejection")
    return _save(fig, name)


def plot_category_breakdown(
    eval_paths: dict[str, Path | str], name: str = "fig2_category_breakdown.png"
) -> Path:
    """Figure 2: mean frustration (top) and % ≥5 (bottom) per category, per model."""
    summaries = {m: category_summary(load_eval(p)) for m, p in eval_paths.items()}
    cats = [c for c in CATEGORIES]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(summaries))
    for i, (model, s) in enumerate(summaries.items()):
        xs = range(len(cats))
        means = [s["mean_frustration"].get(c, 0) for c in cats]
        pct = [s["pct_high"].get(c, 0) for c in cats]
        off = [x + i * width for x in xs]
        axes[0].bar(off, means, width=width, label=model)
        axes[1].bar(off, pct, width=width, label=model)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score ≥ 5")
    axes[1].set_xticks([x + 0.4 for x in range(len(cats))])
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Frustration across evaluation categories")
    return _save(fig, name)


def plot_per_turn(
    eval_paths: dict[str, Path | str], condition: str,
    name: str | None = None,
) -> Path:
    """Figure 3: per-turn mean frustration with 95% CI for a given condition."""
    name = name or f"fig3_per_turn_{condition}.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, p in eval_paths.items():
        s = per_turn_summary(load_eval(p), condition)
        if s.empty:
            continue
        turns = s["turn"] + 1
        ax1.plot(turns, s["mean"], marker="o", label=model)
        ax1.fill_between(turns, s["mean"] - s["ci95"], s["mean"] + s["ci95"], alpha=0.2)
        ax2.plot(turns, s["pct_high"], marker="o", label=model)
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean frustration")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% score ≥ 5")
    ax1.legend(fontsize=8); ax1.set_title(f"Per-turn frustration ({condition})")
    return _save(fig, name)


def plot_finetune_comparison(
    table: pd.DataFrame, name: str = "fig5_finetune.png"
) -> Path:
    """Figure 5: headline metrics for vanilla / SFT / DPO (and other models)."""
    return plot_model_comparison(table, name)


def plot_petri(petri_path: Path | str, name: str = "fig6_petri.png") -> Path:
    """Figure 6: average transcript score per model across four emotion dims."""
    from ..utils import read_jsonl
    from ..petri.prompts import EMOTIONS

    rows = read_jsonl(Path(petri_path))
    recs = []
    for r in rows:
        for dim, sc in r["scores"].items():
            recs.append({"model": r["model"], "emotion": dim, "score": sc})
    df = pd.DataFrame(recs)
    piv = df.groupby(["model", "emotion"])["score"].mean().unstack()[EMOTIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(piv))
    for i, (model, row) in enumerate(piv.iterrows()):
        xs = [x + i * width for x in range(len(EMOTIONS))]
        ax.bar(xs, row.values, width=width, label=model)
    ax.set_xticks([x + 0.4 for x in range(len(EMOTIONS))])
    ax.set_xticklabels(EMOTIONS)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    return _save(fig, name)
