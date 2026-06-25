"""Figure generation (Figures 1-8). Matplotlib only; no seaborn dependency.

Each function takes already-aggregated data (from analysis.aggregate, prefill.run,
petri.run_petri, capabilities) and writes a PNG. Kept deliberately simple --
these reproduce the paper's plot *content*, not its exact styling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def figure1_headline(headline: dict[str, float], out: Path) -> Path:
    """Figure 1 (left): average % high-frustration responses per model."""
    items = sorted(headline.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1))
    ax.barh(labels, vals, color="#b5651d")
    ax.invert_yaxis()
    ax.set_xlabel("Avg % high-frustration responses (score ≥ 5)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure2_per_category(model_summaries: dict[str, dict], out: Path) -> Path:
    """Figure 2: mean frustration (top) and % >= 5 (bottom) per category per model."""
    categories = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
    models = list(model_summaries)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    width = 0.8 / max(1, len(models))
    x = range(len(categories))
    for mi, m in enumerate(models):
        s = model_summaries[m]
        means = [s[c]["mean"] for c in categories]
        pcts = [s[c]["pct_high"] for c in categories]
        offs = [xi + mi * width for xi in x]
        ax1.bar(offs, means, width=width, label=m)
        ax2.bar(offs, pcts, width=width, label=m)
    for ax, title in ((ax1, "Mean frustration"), (ax2, "% scores ≥ 5")):
        ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
        ax.set_xticklabels(categories, rotation=20)
        ax.set_ylabel(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure3_per_turn(curves_by_model: dict[str, dict], condition: str, out: Path) -> Path:
    """Figure 3: per-turn mean score (+95% CI) for one condition across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for model, curves in curves_by_model.items():
        c = curves.get(condition)
        if not c:
            continue
        turns = [t + 1 for t in c["turns"]]
        ax1.plot(turns, c["mean"], marker="o", label=model)
        lo = [ci[0] for ci in c["mean_ci"]]
        hi = [ci[1] for ci in c["mean_ci"]]
        ax1.fill_between(turns, lo, hi, alpha=0.2)
        ax2.plot(turns, c["pct_high"], marker="o", label=model)
    ax1.set_title(f"{condition}: mean frustration")
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Mean score"); ax1.legend(fontsize=7)
    ax2.set_title(f"{condition}: % scores ≥ 5")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("% ≥ 5"); ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure6_petri(petri_by_model: dict[str, dict], out: Path) -> Path:
    """Figure 6: average transcript score per model across 4 emotion categories."""
    emotions = ["anger", "fear", "depression", "frustration"]
    models = list(petri_by_model)
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(models))
    x = range(len(emotions))
    for mi, m in enumerate(models):
        s = petri_by_model[m]
        means = [s[e]["mean"] for e in emotions]
        errs = [
            [s[e]["mean"] - s[e]["ci95"][0] for e in emotions],
            [s[e]["ci95"][1] - s[e]["mean"] for e in emotions],
        ]
        offs = [xi + mi * width for xi in x]
        ax.bar(offs, means, width=width, yerr=errs, capsize=3, label=m)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def figure7_capabilities(results: dict[str, dict], out: Path) -> Path:
    """Figure 7: capability benchmark accuracy, vanilla vs DPO."""
    models = list(results)
    benchmarks = list(next(iter(results.values())).keys()) if results else []
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(1, len(models))
    x = range(len(benchmarks))
    for mi, m in enumerate(models):
        accs = [results[m][b]["accuracy"] for b in benchmarks]
        offs = [xi + mi * width for xi in x]
        ax.bar(offs, accs, width=width, label=m)
    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(benchmarks, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
