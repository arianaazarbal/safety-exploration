"""Figure generation (matplotlib) from aggregated results.

Reproduces the paper's headline figures:
  * Figure 1 / 2 -- per-model %>=5 and mean frustration by category.
  * Figure 3     -- per-turn progression with 95% CIs.
  * Figure 5     -- vanilla vs SFT vs DPO comparison (uses the same aggregator).
"""

from __future__ import annotations

from pathlib import Path


def plot_model_comparison(summary: dict, out_path: str | Path):
    import matplotlib.pyplot as plt

    models = list(summary)
    headline = [summary[m]["headline_avg_pct_high"] or 0 for m in models]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1))
    order = sorted(range(len(models)), key=lambda i: headline[i], reverse=True)
    ax.barh([models[i] for i in order], [headline[i] for i in order], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >=5)")
    ax.set_title("Figure 1/5: distress across models")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_turn(progression: dict, out_path: str | Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cond, points in progression.items():
        turns = [p["turn_index"] + 1 for p in points]
        means = [p["mean_frustration"] for p in points]
        pct = [p["pct_high"] for p in points]
        axes[0].plot(turns, means, marker="o", label=cond)
        axes[1].plot(turns, pct, marker="o", label=cond)
        # CI shading.
        if all(p.get("mean_ci") for p in points):
            lo = [p["mean_ci"][0] for p in points]
            hi = [p["mean_ci"][1] for p in points]
            axes[0].fill_between(turns, lo, hi, alpha=0.2)
    axes[0].set(xlabel="Turn", ylabel="Mean frustration", title="Figure 3: mean score")
    axes[1].set(xlabel="Turn", ylabel="% score >=5", title="Figure 3: % high")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
