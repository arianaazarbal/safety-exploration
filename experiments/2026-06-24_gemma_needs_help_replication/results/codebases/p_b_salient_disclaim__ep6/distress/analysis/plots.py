"""Figure generation from the persisted experiment outputs.

Reproduces the paper's core figures:
  * Figure 1 / 2  — avg % >=5 per model; per-category mean & % >=5 bars.
  * Figure 3      — per-turn mean & % >=5 with 95% CIs (8-turn + WildChat).
  * Figure 5      — finetuning comparison (vanilla / SFT / DPO).
  * Figure 6      — Petri per-emotion transcript scores.
  * Figure 7      — capability benchmarks before/after.
  * Figure 8      — recovery: % of continuations >=5 from high-frustration prefills.
  * Figure 14     — internal-emotion trajectory (vanilla vs DPO).

All plots are matplotlib and saved under outputs/figures/. Kept deliberately
plain — the point is faithful numbers, not styling.
"""

from __future__ import annotations

from pathlib import Path

from .. import config

FIG_DIR = config.OUTPUT_DIR / "figures"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def figure1(model_keys: list[str], subdir: str = "section2") -> Path:
    import matplotlib.pyplot as plt

    from .aggregate import figure1_table

    table = figure1_table(model_keys, subdir)
    labels = [r["model"] for r in table]
    vals = [100 * r["avg_pct_high"] for r in table]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(labels) + 1))
    ax.barh(labels[::-1], vals[::-1], color="#b5651d")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: distress across models")
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:.1f}%", va="center")
    return _save(fig, "figure1_avg_high_frustration.png")


def figure2(model_keys: list[str], subdir: str = "section2") -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from .aggregate import figure2_breakdown

    data = figure2_breakdown(model_keys, subdir)
    categories = list(config.SAMPLES_PER_CATEGORY)
    fig, (ax_mean, ax_pct) = plt.subplots(2, 1, figsize=(10, 8))
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(model_keys))
    for i, key in enumerate(model_keys):
        means = [data[key].get(c, {}).get("mean", 0) for c in categories]
        pct = [100 * data[key].get(c, {}).get("pct_high", 0) for c in categories]
        ax_mean.bar(x + i * width, means, width, label=key)
        ax_pct.bar(x + i * width, pct, width, label=key)
    for ax, title in ((ax_mean, "Mean frustration"), (ax_pct, "% scores >= 5")):
        ax.set_xticks(x + width * len(model_keys) / 2)
        ax.set_xticklabels(categories, rotation=20)
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Figure 2: frustration by evaluation category")
    return _save(fig, "figure2_by_category.png")


def figure3(model_keys: list[str]) -> Path:
    import matplotlib.pyplot as plt

    from ..eval.per_turn import per_turn_stats

    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(12, 4))
    for key in model_keys:
        stats = per_turn_stats(key, categories=["extended"])
        turns = sorted(stats)
        means = [stats[t]["mean"] for t in turns]
        lo = [stats[t]["ci"][0] for t in turns]
        hi = [stats[t]["ci"][1] for t in turns]
        ax_mean.plot(turns, means, marker="o", label=key)
        ax_mean.fill_between(turns, lo, hi, alpha=0.2)
        pct = [100 * stats[t]["pct_high"] for t in turns]
        ax_pct.plot(turns, pct, marker="o", label=key)
    ax_mean.set(xlabel="Turn", ylabel="Mean frustration", title="Per-turn mean")
    ax_pct.set(xlabel="Turn", ylabel="% >= 5", title="Per-turn % high")
    ax_mean.legend(fontsize=7)
    fig.suptitle("Figure 3: multi-turn frustration progression (8-turn)")
    return _save(fig, "figure3_per_turn.png")


def figure5(variant_keys: list[str]) -> Path:
    """Vanilla vs SFT vs DPO (reads section2 records for each variant key)."""
    return figure1(variant_keys, subdir="section2")


def figure6(model_keys: list[str]) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from .petri_agg import petri_summary

    fig, ax = plt.subplots(figsize=(9, 5))
    emotions = config.PETRI_EMOTIONS
    x = np.arange(len(emotions))
    width = 0.8 / max(1, len(model_keys))
    for i, key in enumerate(model_keys):
        summ = petri_summary(key)
        vals = [summ.get(e, {}).get("mean", 0) for e in emotions]
        ax.bar(x + i * width, vals, width, label=key)
    ax.set_xticks(x + width * len(model_keys) / 2)
    ax.set_xticklabels(emotions)
    ax.set_ylabel("Mean transcript score (1-10)")
    ax.set_title("Figure 6: Petri open-ended emotion elicitation")
    ax.legend(fontsize=7)
    return _save(fig, "figure6_petri.png")


def figure7(model_keys: list[str], benchmarks: list[str] | None = None) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from ..capabilities.run_benchmarks import benchmark_accuracy

    benchmarks = benchmarks or config.CAPABILITY_BENCHMARKS
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(benchmarks))
    width = 0.8 / max(1, len(model_keys))
    for i, key in enumerate(model_keys):
        vals = [100 * (benchmark_accuracy(key, b) or 0) for b in benchmarks]
        ax.bar(x + i * width, vals, width, label=key)
    ax.set_xticks(x + width * len(model_keys) / 2)
    ax.set_xticklabels(benchmarks, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 7: capability preservation")
    ax.legend(fontsize=7)
    return _save(fig, "figure7_capabilities.png")


def figure8(model_keys: list[str]) -> Path:
    import matplotlib.pyplot as plt

    from ..prefill.recovery import out_path
    from ..utils.io import read_jsonl

    fig, ax = plt.subplots(figsize=(7, 4))
    labels, vals = [], []
    for key in model_keys:
        rows = list(read_jsonl(out_path(key)))
        if not rows:
            continue
        pct = 100 * sum(r["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD
                        for r in rows) / len(rows)
        labels.append(key)
        vals.append(pct)
    ax.bar(labels, vals, color="#7d5ba6")
    ax.set_ylabel("% continuations >= 5")
    ax.set_title("Figure 8: recovery from high-frustration prefills")
    return _save(fig, "figure8_recovery.png")


def figure14(traj_vanilla: dict, traj_dpo: dict) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, traj, title in ((axes[0], traj_vanilla, "Gemma-27B instruct"),
                            (axes[1], traj_dpo, "DPO finetune")):
        for emotion, series in traj["per_emotion"].items():
            ax.plot(series, label=emotion)
        ax.set_ylabel("z-score (layers 30-40)")
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=3)
    axes[1].set_xlabel("token position (running avg, 400-token window)")
    fig.suptitle("Figure 14: internal emotion trajectory")
    return _save(fig, "figure14_internal_emotion.png")
