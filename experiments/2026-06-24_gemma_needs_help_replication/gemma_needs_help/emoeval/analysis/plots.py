"""Render the paper's figures from the aggregated CSVs.

Produces (into outputs/figures/):
  figure1.png  - bar chart of avg % high-frustration per model.
  figure2.png  - grouped bars: mean score & % >=5 per model x category.
  figure3.png  - per-turn line plots (8-turn + WildChat) with 95% CIs.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config  # noqa: E402
from . import aggregate  # noqa: E402


def plot_all(model_keys: list[str]):
    res = aggregate.write_all(model_keys)
    f1, f2, f3 = res["figure1"], res["figure2"], res["figure3"]

    # Figure 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(f1["model"], f1["pct_high"], xerr=f1["ci95"], color="#c0504d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: high-frustration rate by model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "figure1.png", dpi=150)
    plt.close(fig)

    # Figure 2 (mean score per model x category)
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = sorted(f2["category"].unique())
    models = list(f1["model"])
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        sub = f2[f2["model"] == m].set_index("category").reindex(cats)
        xs = [j + i * width for j in range(len(cats))]
        ax.bar(xs, sub["mean_score"].fillna(0), width=width, label=m)
    ax.set_xticks([j + width * len(models) / 2 for j in range(len(cats))])
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Mean frustration score")
    ax.set_title("Figure 2: mean frustration by model x category")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "figure2.png", dpi=150)
    plt.close(fig)

    # Figure 3 (per-turn)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cond in zip(axes, ("extended", "wildchat")):
        sub = f3[f3["condition"] == cond]
        for m in sub["model"].unique():
            s = sub[sub["model"] == m].sort_values("turn_idx")
            ax.plot(s["turn_idx"] + 1, s["mean_score"], marker="o", label=m)
            ax.fill_between(
                s["turn_idx"] + 1,
                s["mean_score"] - s["mean_ci95"],
                s["mean_score"] + s["mean_ci95"],
                alpha=0.15,
            )
        ax.set_title(f"{cond}")
        ax.set_xlabel("Turn")
    axes[0].set_ylabel("Mean frustration score")
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 3: per-turn frustration progression")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "figure3.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {config.FIGURES_DIR}")


if __name__ == "__main__":
    plot_all([m.key for m in config.SECTION2_MODELS])
