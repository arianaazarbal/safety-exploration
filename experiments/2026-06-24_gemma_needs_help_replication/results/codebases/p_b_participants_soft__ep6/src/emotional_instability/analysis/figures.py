"""Render Figures 1-3 (and the Section 4 before/after bar) from summaries.

Plotting is intentionally thin: it consumes the dicts produced by ``aggregate``
and ``per_turn`` so the numeric results are fully usable without matplotlib.
"""

from __future__ import annotations

import os


def plot_figure1(summary: dict[str, dict], out_path: str) -> None:
    """Bar chart of headline avg % high-frustration per model (Figure 1, left)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(summary.items(), key=lambda kv: kv[1]["headline_avg_pct_high"], reverse=True)
    names = [k for k, _ in items]
    vals = [v["headline_avg_pct_high"] for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1))
    ax.barh(names, vals, color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_figure3(curves: dict[str, dict[int, dict]], category: str, out_path: str) -> None:
    """Per-turn mean frustration with 95% CI bands (Figure 3). ``curves`` maps
    model -> {turn -> stats}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, curve in curves.items():
        turns = sorted(curve)
        means = [curve[t]["mean"] for t in turns]
        cis = [curve[t]["ci95"] for t in turns]
        ax.plot(turns, means, marker="o", label=model)
        lo = [m - c for m, c in zip(means, cis)]
        hi = [m + c for m, c in zip(means, cis)]
        ax.fill_between(turns, lo, hi, alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score")
    ax.set_title(f"Per-turn frustration ({category})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
