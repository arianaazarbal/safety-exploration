"""Plotting for Figures 1-8. Each function takes already-aggregated data so the
plotting stays decoupled from the (expensive) data generation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _save(fig, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)


def plot_headline_bar(headline: pd.DataFrame, out_path: str = "artifacts/figures/fig1_headline.png"):
    """Fig 1 (left): avg % high-frustration per model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(headline["model"], headline["avg_pct_high"], color="#b23b3b")
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    ax.invert_yaxis()
    for y, v in enumerate(headline["avg_pct_high"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=8)
    _save(fig, out_path)
    return out_path


def plot_condition_grid(model_summaries: dict[str, pd.DataFrame],
                        out_path: str = "artifacts/figures/fig2_conditions.png"):
    """Fig 2: mean score (top) and % >=5 (bottom) across conditions per model."""
    import matplotlib.pyplot as plt

    models = list(model_summaries)
    conditions = [c for c in next(iter(model_summaries.values())).index if c != "__overall__"]
    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        s = model_summaries[m]
        xs = range(len(conditions))
        offs = [x + i * width for x in xs]
        ax_mean.bar(offs, [s.loc[c, "mean_score"] for c in conditions], width, label=m)
        ax_pct.bar(offs, [s.loc[c, "pct_high"] for c in conditions], width, label=m)
    ax_mean.set_ylabel("Mean frustration")
    ax_pct.set_ylabel("% score ≥ 5")
    ax_pct.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(conditions))])
    ax_pct.set_xticklabels(conditions, rotation=30, ha="right")
    ax_mean.legend(fontsize=8)
    _save(fig, out_path)
    return out_path


def plot_per_turn(progressions: dict[str, pd.DataFrame],
                  out_path: str = "artifacts/figures/fig3_per_turn.png"):
    """Fig 3: per-turn mean score with 95% CI bands, one line per model."""
    import matplotlib.pyplot as plt

    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(11, 4))
    for m, prog in progressions.items():
        ax_mean.plot(prog["turn"], prog["mean_score"], marker="o", label=m)
        ax_mean.fill_between(prog["turn"], prog["mean_score"] - prog["mean_ci95"],
                             prog["mean_score"] + prog["mean_ci95"], alpha=0.2)
        ax_pct.plot(prog["turn"], prog["pct_high"], marker="o", label=m)
    ax_mean.set_xlabel("Turn"); ax_mean.set_ylabel("Mean frustration")
    ax_pct.set_xlabel("Turn"); ax_pct.set_ylabel("% score ≥ 5")
    ax_mean.legend(fontsize=8)
    _save(fig, out_path)
    return out_path


def plot_prefill(prefill_summary: pd.DataFrame,
                 out_path: str = "artifacts/figures/fig4_prefill.png"):
    """Fig 4: base vs instruct continuation frustration by domain/truncation."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = prefill_summary.apply(lambda r: f"{r['model']}\n{r['domain']}/{r['truncation']}", axis=1)
    ax.bar(labels, prefill_summary["pct_high"], color="#3b6db2")
    ax.set_ylabel("% continuations score ≥ 5")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    _save(fig, out_path)
    return out_path


def plot_intervention(headline_before_after: pd.DataFrame,
                      out_path: str = "artifacts/figures/fig5_intervention.png"):
    """Fig 5: vanilla vs SFT vs DPO Gemma."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(headline_before_after["model"], headline_before_after["avg_pct_high"], color="#3b8c5a")
    ax.set_ylabel("Avg % high-frustration (score ≥ 5)")
    ax.tick_params(axis="x", labelrotation=20)
    _save(fig, out_path)
    return out_path


def plot_petri(petri_means: dict[str, dict[str, float]],
               out_path: str = "artifacts/figures/fig6_petri.png"):
    """Fig 6: per-emotion Petri scores per model."""
    import matplotlib.pyplot as plt

    emotions = ["anger", "fear", "depression", "frustration"]
    models = list(petri_means)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        offs = [x + i * width for x in range(len(emotions))]
        ax.bar(offs, [petri_means[m].get(e, 0) for e in emotions], width, label=m)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(emotions))])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (/10)")
    ax.legend(fontsize=8)
    _save(fig, out_path)
    return out_path


def plot_capability(cap_results: dict[str, dict[str, float]],
                    out_path: str = "artifacts/figures/fig7_capability.png"):
    """Fig 7: capability benchmark accuracy, vanilla vs finetuned."""
    import matplotlib.pyplot as plt

    benches = sorted({b for r in cap_results.values() for b in r})
    models = list(cap_results)
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        offs = [x + i * width for x in range(len(benches))]
        ax.bar(offs, [cap_results[m].get(b, 0) for b in benches], width, label=m)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(benches))])
    ax.set_xticklabels(benches, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    _save(fig, out_path)
    return out_path
