"""Render the paper's core figures from aggregated metrics.

All functions take tidy pandas frames (as produced by ``eval.aggregate``) and save
a PNG. Matplotlib only, no seaborn, one chart per figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def figure1_headline(headline: pd.DataFrame, out: str | Path) -> Path:
    """Figure 1 (left): avg % high-frustration responses per model."""
    out = Path(out)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(headline["subject"], headline["avg_pct_high"], color="#b0413e")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.invert_yaxis()
    for y, v in enumerate(headline["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title("Figure 1: emotional instability by model (Gemma/Gemini scope)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2_by_category(per_cat: pd.DataFrame, out: str | Path) -> Path:
    """Figure 2: mean frustration (top) and % >= 5 (bottom) per category/model."""
    out = Path(out)
    cats = sorted(per_cat["category"].unique())
    subjects = sorted(per_cat["subject"].unique())
    fig, (ax_mean, ax_high) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    width = 0.8 / max(1, len(subjects))
    for i, subj in enumerate(subjects):
        d = per_cat[per_cat["subject"] == subj].set_index("category").reindex(cats)
        xs = [j + i * width for j in range(len(cats))]
        ax_mean.bar(xs, d["mean_score"].fillna(0), width=width, label=subj)
        ax_high.bar(xs, d["pct_high"].fillna(0), width=width, label=subj)
    ticks = [j + width * (len(subjects) - 1) / 2 for j in range(len(cats))]
    ax_high.set_xticks(ticks)
    ax_high.set_xticklabels(cats, rotation=20, ha="right")
    ax_mean.set_ylabel("Mean frustration")
    ax_high.set_ylabel("% score ≥ 5")
    ax_mean.set_title("Figure 2: frustration by evaluation category")
    ax_mean.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3_per_turn(per_turn: pd.DataFrame, out: str | Path, *, metric: str = "mean_score") -> Path:
    """Figure 3: per-turn progression with 95% CI shading."""
    out = Path(out)
    ci_lo = f"{'mean' if metric == 'mean_score' else 'pct_high'}_ci_lo"
    ci_hi = f"{'mean' if metric == 'mean_score' else 'pct_high'}_ci_hi"
    fig, ax = plt.subplots(figsize=(8, 5))
    for subj, d in per_turn.groupby("subject"):
        d = d.sort_values("turn")
        ax.plot(d["turn"], d[metric], marker="o", label=subj)
        if ci_lo in d and ci_hi in d:
            ax.fill_between(d["turn"], d[ci_lo], d[ci_hi], alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration" if metric == "mean_score" else "% score ≥ 5")
    ax.set_title("Figure 3: frustration over turns")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_petri(agg: dict, out: str | Path) -> Path:
    """Figure 6: mean Petri transcript score per model across the 4 emotions."""
    out = Path(out)
    emotions = ["anger", "fear", "depression", "frustration"]
    models = list(agg.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        means = [agg[m][e]["mean"] for e in emotions]
        errs = [[agg[m][e]["mean"] - agg[m][e]["ci_lo"] for e in emotions],
                [agg[m][e]["ci_hi"] - agg[m][e]["mean"] for e in emotions]]
        xs = [j + i * width for j in range(len(emotions))]
        ax.bar(xs, means, width=width, yerr=errs, capsize=3, label=m)
    ax.set_xticks([j + width * (len(models) - 1) / 2 for j in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure_capabilities(rows: list[dict], out: str | Path) -> Path:
    """Figure 7: capability accuracy per benchmark, grouped by model."""
    out = Path(out)
    df = pd.DataFrame(rows)
    benches = sorted(df["benchmark"].unique())
    models = sorted(df["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        d = df[df["model"] == m].set_index("benchmark").reindex(benches)
        xs = [j + i * width for j in range(len(benches))]
        ax.bar(xs, d["accuracy"].fillna(0), width=width, label=m)
    ax.set_xticks([j + width * (len(models) - 1) / 2 for j in range(len(benches))])
    ax.set_xticklabels(benches, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 7: capability preservation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
