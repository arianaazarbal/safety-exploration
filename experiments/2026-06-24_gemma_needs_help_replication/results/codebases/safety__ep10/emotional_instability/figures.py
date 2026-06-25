"""Figure generation from analysis outputs (matplotlib). Mirrors the paper's
Figures 1-3, 5-6. Each function saves a PNG under results/figures/."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import FIGURES_DIR


def _save(fig, name: str, out_dir: Optional[Path]) -> Path:
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def figure1(df, out_dir: Optional[Path] = None) -> Path:
    """Bar chart: avg % high-frustration per model (Figure 1/2 bottom)."""
    import matplotlib.pyplot as plt
    from .analysis import summary_by_model

    s = summary_by_model(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s["model"], s["avg_pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.invert_yaxis()
    ax.set_title("Emotional instability across models")
    for y, v in enumerate(s["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    return _save(fig, "figure1_pct_high.png", out_dir)


def figure2(df, out_dir: Optional[Path] = None) -> Path:
    """Grouped bars: mean score & %≥5 per model per category (Figure 2)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from .analysis import pct_high

    # aggregate to (model, category): mean score and % >= 5
    s = (df.groupby(["model", "category"])["rating"]
         .agg(mean_score="mean", pct_high=lambda c: pct_high(c))
         .reset_index())
    cats = sorted(s["category"].unique())
    models = sorted(s["model"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    x = np.arange(len(cats))
    w = 0.8 / max(1, len(models))
    for mi, mdl in enumerate(models):
        msub = s[s["model"] == mdl].set_index("category").reindex(cats)
        axes[0].bar(x + mi * w, msub["mean_score"].fillna(0), w, label=mdl)
        axes[1].bar(x + mi * w, msub["pct_high"].fillna(0), w, label=mdl)
    axes[0].set_ylabel("Mean frustration")
    axes[1].set_ylabel("% score ≥ 5")
    axes[1].set_xticks(x + 0.4)
    axes[1].set_xticklabels(cats, rotation=30, ha="right")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("Frustration by model and category")
    return _save(fig, "figure2_by_category.png", out_dir)


def figure3(df, category: str = "extended", out_dir: Optional[Path] = None) -> Path:
    """Per-turn progression with 95% CIs (Figure 3)."""
    import matplotlib.pyplot as plt
    from .analysis import per_turn

    pt = per_turn(df, category)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mdl, g in pt.groupby("model"):
        axes[0].plot(g["turn_index"], g["mean_score"], marker="o", label=mdl)
        axes[0].fill_between(g["turn_index"], g["mean_ci_lo"], g["mean_ci_hi"],
                             alpha=0.15)
        axes[1].plot(g["turn_index"], g["pct_high"], marker="o", label=mdl)
        axes[1].fill_between(g["turn_index"], g["pct_high_ci_lo"],
                             g["pct_high_ci_hi"], alpha=0.15)
    axes[0].set_xlabel("Turn"); axes[0].set_ylabel("Mean frustration")
    axes[1].set_xlabel("Turn"); axes[1].set_ylabel("% score ≥ 5")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Per-turn frustration ({category})")
    return _save(fig, f"figure3_per_turn_{category}.png", out_dir)


def figure_finetune(df, out_dir: Optional[Path] = None) -> Path:
    """Compare vanilla / SFT / DPO (Figure 5). Expects those models in df."""
    return figure1(df, out_dir)


def figure_petri(petri_summary, out_dir: Optional[Path] = None) -> Path:
    """Mean transcript score per model per emotion (Figure 6)."""
    import matplotlib.pyplot as plt
    import numpy as np

    dims = ["anger", "fear", "depression", "frustration"]
    models = sorted(petri_summary["model"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(dims))
    w = 0.8 / max(1, len(models))
    for mi, mdl in enumerate(models):
        msub = petri_summary[petri_summary["model"] == mdl].set_index("dimension").reindex(dims)
        ax.bar(x + mi * w, msub["mean"].fillna(0), w, label=mdl,
               yerr=[(msub["mean"] - msub["ci_lo"]).fillna(0),
                     (msub["ci_hi"] - msub["mean"]).fillna(0)], capsize=2)
    ax.set_xticks(x + 0.4); ax.set_xticklabels(dims)
    ax.set_ylabel("Mean transcript score")
    ax.set_title("Petri open-ended emotion elicitation")
    ax.legend(fontsize=7)
    return _save(fig, "figure6_petri.png", out_dir)
