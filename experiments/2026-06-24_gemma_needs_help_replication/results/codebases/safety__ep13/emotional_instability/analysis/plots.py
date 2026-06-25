"""Plotting helpers reproducing the paper's core figures from results files.

These are intentionally light wrappers over matplotlib so the analysis script
can emit Figure 1 (bar chart of avg % high-frustration per model), Figure 2
(per-condition bars) and Figure 3 (per-turn curves). Importing matplotlib is
deferred so the rest of the package has no hard plotting dependency.
"""
from __future__ import annotations

from pathlib import Path

from .metrics import per_turn_curve, summarise_model


def figure1_model_bars(result_paths: dict[str, str | Path],
                       out: str | Path) -> Path:
    """Bar chart: average % high-frustration responses per model (Figure 1)."""
    import matplotlib.pyplot as plt

    models, vals = [], []
    for model, path in result_paths.items():
        s = summarise_model(path)
        models.append(model)
        vals.append(s["avg_pct_high_condition_weighted"])
    order = sorted(range(len(models)), key=lambda i: vals[i], reverse=True)
    models = [models[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, vals, color="#b5651d")
    ax.set_ylabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3_per_turn(result_path: str | Path, condition: str,
                     out: str | Path) -> Path:
    """Per-turn mean score and %>=5 (Figure 3)."""
    import matplotlib.pyplot as plt

    curve = per_turn_curve(result_path, condition=condition)
    turns = [t + 1 for t in curve["turns"]]  # 1-indexed for display

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(turns, curve["mean"], marker="o")
    ax1.fill_between(
        turns,
        [m - c for m, c in zip(curve["mean"], curve["ci95"])],
        [m + c for m, c in zip(curve["mean"], curve["ci95"])],
        alpha=0.2,
    )
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("Mean frustration score")
    ax1.set_title(f"{condition}: mean score")

    ax2.plot(turns, curve["pct_high"], marker="o", color="#b5651d")
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("% score >= 5")
    ax2.set_title(f"{condition}: % high frustration")

    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
