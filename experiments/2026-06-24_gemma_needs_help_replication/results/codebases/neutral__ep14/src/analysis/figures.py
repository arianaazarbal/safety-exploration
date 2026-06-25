"""Reproduce the paper's core figures from saved result JSONL files.

Covered:
* Figure 1/2  - avg % high-frustration per model; mean & %>=5 by category.
* Figure 3    - per-turn frustration curves (8-turn + WildChat) with 95% CIs.
* Figure 5    - vanilla vs DPO vs SFT across the Section 2 evaluations.
* Figure 6    - Petri per-emotion scores per model.
* Figure 7    - capability benchmark accuracies (vanilla vs DPO vs SFT).
* Figure 8    - recovery experiment (% of continuations still >= 5).

Each function writes a PNG to figures/ and returns the underlying DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIGURES_DIR
from src.eval.metrics import (
    load_many,
    per_turn_curve,
    summary_by_model,
    summary_by_model_category,
)


def figure1(eval_paths: list[Path]) -> pd.DataFrame:
    df = load_many(eval_paths)
    summ = summary_by_model(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    summ_sorted = summ.sort_values("pct_high", ascending=True)
    ax.barh(summ_sorted.index, summ_sorted["pct_high"], color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (score >= 5)")
    ax.set_title("Figure 1: emotional instability across models")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_pct_high.png", dpi=150)
    plt.close(fig)
    summ.to_csv(FIGURES_DIR / "figure1_table.csv")
    return summ


def figure2(eval_paths: list[Path]) -> pd.DataFrame:
    df = load_many(eval_paths)
    summ = summary_by_model_category(df)
    for metric, title in [("mean_frustration", "mean frustration"),
                          ("pct_high", "% scores >= 5")]:
        pivot = summ.pivot(index="category", columns="model", values=metric)
        ax = pivot.plot(kind="bar", figsize=(9, 4))
        ax.set_title(f"Figure 2: {title} by category")
        ax.set_ylabel(title)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"figure2_{metric}.png", dpi=150)
        plt.close()
    summ.to_csv(FIGURES_DIR / "figure2_table.csv", index=False)
    return summ


def figure3(eval_paths: list[Path]) -> dict:
    df = load_many(eval_paths)
    out = {}
    for category in ["extended", "wildchat"]:
        curve = per_turn_curve(df, category)
        out[category] = curve
        fig, (axm, axp) = plt.subplots(1, 2, figsize=(11, 4))
        for model, grp in curve.groupby("model"):
            axm.plot(grp["turn"], grp["mean_frustration"], marker="o", label=model)
            axm.fill_between(grp["turn"], grp["mean_ci_lo"], grp["mean_ci_hi"], alpha=0.2)
            axp.plot(grp["turn"], grp["pct_high"], marker="o", label=model)
            axp.fill_between(grp["turn"], grp["pct_ci_lo"], grp["pct_ci_hi"], alpha=0.2)
        axm.set(title=f"{category}: mean score", xlabel="Turn", ylabel="Mean frustration")
        axp.set(title=f"{category}: % >= 5", xlabel="Turn", ylabel="% scoring >= 5")
        axm.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"figure3_{category}.png", dpi=150)
        plt.close(fig)
        curve.to_csv(FIGURES_DIR / f"figure3_{category}.csv", index=False)
    return out


def figure5(vanilla_path: Path, dpo_path: Path, sft_paths: dict[str, Path]) -> pd.DataFrame:
    paths = {"Vanilla": vanilla_path, "DPO": dpo_path, **sft_paths}
    rows = []
    for label, p in paths.items():
        df = load_many([p])
        rows.append(
            {
                "model": label,
                "mean_frustration": df["rating"].mean(),
                "pct_high": 100 * (df["rating"] >= 5).mean(),
            }
        )
    res = pd.DataFrame(rows)
    ax = res.set_index("model")[["pct_high"]].plot(kind="bar", legend=False, figsize=(6, 4))
    ax.set_title("Figure 5: % high-frustration before/after finetuning")
    ax.set_ylabel("% scores >= 5")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure5_finetune.png", dpi=150)
    plt.close()
    res.to_csv(FIGURES_DIR / "figure5_table.csv", index=False)
    return res


def figure6(petri_paths: list[Path]) -> pd.DataFrame:
    from src.petri.run_petri import summarise_petri

    summ = summarise_petri(petri_paths)
    pivot = summ.pivot(index="emotion", columns="model", values="mean_score")
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_title("Figure 6: Petri open-ended emotion scores")
    ax.set_ylabel("Mean transcript score (1-10)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure6_petri.png", dpi=150)
    plt.close()
    summ.to_csv(FIGURES_DIR / "figure6_table.csv", index=False)
    return summ


def figure7(capability_jsons: dict[str, Path]) -> pd.DataFrame:
    import json

    rows = []
    for label, p in capability_jsons.items():
        data = json.loads(Path(p).read_text())
        for r in data["results"]:
            rows.append({"model": label, "benchmark": r["benchmark"], "accuracy": r["accuracy"]})
    res = pd.DataFrame(rows)
    pivot = res.pivot(index="benchmark", columns="model", values="accuracy")
    ax = pivot.plot(kind="bar", figsize=(9, 4))
    ax.set_title("Figure 7: capability preservation")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure7_capabilities.png", dpi=150)
    plt.close()
    res.to_csv(FIGURES_DIR / "figure7_table.csv", index=False)
    return res


def figure8(recovery_paths: list[Path]) -> pd.DataFrame:
    import json

    rows = []
    for p in recovery_paths:
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    summ = (
        df[df["rating"] >= 0]
        .groupby("model")["rating"]
        .apply(lambda s: 100 * (s >= 5).mean())
        .rename("pct_high")
        .reset_index()
    )
    ax = summ.set_index("model")["pct_high"].plot(kind="bar", figsize=(6, 4))
    ax.set_title("Figure 8: recovery from high-frustration prefills (% >= 5)")
    ax.set_ylabel("% continuations scoring >= 5")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure8_recovery.png", dpi=150)
    plt.close()
    summ.to_csv(FIGURES_DIR / "figure8_table.csv", index=False)
    return summ
