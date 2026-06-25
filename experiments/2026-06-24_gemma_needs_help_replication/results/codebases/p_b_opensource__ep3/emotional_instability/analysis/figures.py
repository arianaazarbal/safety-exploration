"""Figure generation from aggregated results (Figures 1-7).

Thin matplotlib wrappers that take the aggregate dicts produced by the runners
and write PNGs to ``config.ARTIFACTS_DIR``. matplotlib is imported lazily so the
rest of the package has no plotting dependency. Each function returns the saved
path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import config


def _save(fig, name: str) -> Path:
    out = config.ARTIFACTS_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out


def plot_model_high_rates(
    high_rate_by_model: Mapping[str, float],
    *,
    name: str = "figure1_high_rates.png",
    title: str = "Average % high-frustration responses (score >= 5)",
):
    """Figure 1 (left): per-model average high-frustration rate, sorted desc."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(high_rate_by_model.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1))
    ax.barh(labels, values, color="#b5341f")
    ax.invert_yaxis()
    ax.set_xlabel("% responses with frustration >= 5")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1f}%", va="center")
    return _save(fig, name)


def plot_per_turn(
    per_turn: Sequence[dict],
    *,
    name: str = "figure3_per_turn.png",
    title: str = "Mean frustration by turn",
):
    """Figure 3: per-turn mean frustration with 95% CI band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    turns = [d["turn"] for d in per_turn]
    means = [d["mean"] for d in per_turn]
    lo = [d["mean_ci"][0] for d in per_turn]
    hi = [d["mean_ci"][1] for d in per_turn]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(turns, means, marker="o", color="#b5341f")
    ax.fill_between(turns, lo, hi, alpha=0.2, color="#b5341f")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration (0-10)")
    ax.set_title(title)
    ax.set_ylim(0, 10)
    return _save(fig, name)


def plot_finetuning_comparison(
    high_rate_by_variant: Mapping[str, float],
    *,
    name: str = "figure5_finetuning.png",
    title: str = "Finetuning effect on high-frustration rate",
):
    """Figure 5: vanilla vs SFT vs DPO high-frustration rates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(high_rate_by_variant)
    values = [high_rate_by_variant[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#3b6ea5")
    ax.set_ylabel("% responses with frustration >= 5")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    return _save(fig, name)


def plot_petri(
    petri_by_model: Mapping[str, Mapping[str, float]],
    *,
    name: str = "figure6_petri.png",
    title: str = "Open-ended emotion elicitation (mean transcript score)",
):
    """Figure 6: grouped bars of per-emotion mean transcript scores per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    emotions = list(config.PETRI.emotions)
    models = list(petri_by_model)
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(models):
        vals = [petri_by_model[m].get(e, 0.0) for e in emotions]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (0-10)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_capabilities(
    accuracy_by_model: Mapping[str, Mapping[str, float]],
    *,
    name: str = "figure7_capabilities.png",
    title: str = "Capability preservation",
):
    """Figure 7: grouped capability accuracies per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    benchmarks = sorted({b for m in accuracy_by_model.values() for b in m})
    models = list(accuracy_by_model)
    x = np.arange(len(benchmarks))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(models):
        vals = [accuracy_by_model[m].get(b, 0.0) for b in benchmarks]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benchmarks, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, name)
