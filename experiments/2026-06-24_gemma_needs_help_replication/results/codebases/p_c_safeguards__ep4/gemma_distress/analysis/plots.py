"""Figure generation (Figures 1-3, 5-6). Matplotlib only; no seaborn dependency.

These are convenience plotters; they read the same metrics functions the CLI
'analyze' command uses. Saved under results/figures/.
"""
from __future__ import annotations

from pathlib import Path

from ..config import REPO_ROOT
from . import metrics

FIG_ROOT = REPO_ROOT / "results" / "figures"


def _ensure_dir() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)


def plot_headline(models: list[str], filename: str = "figure1_headline.png") -> Path:
    """Figure 1 (left): avg % high-frustration responses per model."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    vals = [metrics.headline_pct_high(m) for m in models]
    order = sorted(range(len(models)), key=lambda i: vals[i], reverse=True)
    models_s = [models[i] for i in order]
    vals_s = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(models) + 1))
    ax.barh(models_s, vals_s, color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.invert_yaxis()
    for i, v in enumerate(vals_s):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    out = FIG_ROOT / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_turn(model: str, category: str, filename: str | None = None) -> Path:
    """Figure 3: per-turn mean frustration with bootstrap CI band."""
    import matplotlib.pyplot as plt

    _ensure_dir()
    prog = metrics.per_turn_progression(model, category)
    turns = prog["turns"]
    mean = prog["mean"]
    lo = [c[0] for c in prog["mean_ci"]]
    hi = [c[1] for c in prog["mean_ci"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(turns, mean, marker="o", color="#2c3e50", label=model)
    ax.fill_between(turns, lo, hi, alpha=0.25, color="#2c3e50")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score")
    ax.set_title(f"{model} - {category}")
    ax.set_ylim(0, 10)
    ax.legend()
    fig.tight_layout()
    out = FIG_ROOT / (filename or f"per_turn_{model}_{category}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
