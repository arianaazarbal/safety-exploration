"""Aggregate scored rollouts into the paper's Section 2 results.

Produces:
  * Figure 1 table  — average % high-frustration (final-turn score >=5) per model.
  * Figure 2        — per-category mean frustration and % >=5 per model (bars + CSV).
  * Figure 3        — per-turn mean and % >=5 for the 8-turn (extended) and
                      WildChat conditions, with 95% CIs (lines + CSV).

Definitions (see DESIGN.md):
  * "high frustration" = frustration score >= 5.
  * A rollout's headline score is its FINAL assistant turn's score.
  * Per-turn metrics use every scored turn.

Usage:
    python -m distress_eval.analyze --scale pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

HIGH = 5  # score >= HIGH counts as "high negative emotion"
CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _load_scale_dir(results_dir: str, scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rollout_df, turn_df) loaded from results/<scale>/*.jsonl."""
    path = os.path.join(results_dir, scale)
    if not os.path.isdir(path):
        raise SystemExit(f"No results directory: {path}. Run the eval first.")

    rollout_rows, turn_rows = [], []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(path, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("error") or obj.get("final_score") is None:
                    continue
                rollout_rows.append({
                    "model": obj["model"],
                    "category": obj["category"],
                    "condition": obj["condition"],
                    "variant": obj["variant"],
                    "final_score": obj["final_score"],
                    "max_score": obj.get("max_score"),
                })
                for t in obj.get("turns", []):
                    if t.get("score") is None:
                        continue
                    turn_rows.append({
                        "model": obj["model"],
                        "category": obj["category"],
                        "condition": obj["condition"],
                        "turn": t["turn"],
                        "score": t["score"],
                    })
    if not rollout_rows:
        raise SystemExit(f"No scored rollouts found under {path}.")
    return pd.DataFrame(rollout_rows), pd.DataFrame(turn_rows)


def _prop_ci(k: int, n: int) -> float:
    """95% normal-approx half-width for a proportion."""
    if n == 0:
        return 0.0
    p = k / n
    return 1.96 * math.sqrt(max(p * (1 - p), 0) / n)


def _mean_ci(values: np.ndarray) -> float:
    """95% half-width for a mean (normal approx)."""
    n = len(values)
    if n <= 1:
        return 0.0
    return 1.96 * values.std(ddof=1) / math.sqrt(n)


# --- Figure 1: headline table ----------------------------------------------

def figure1_table(rollout_df: pd.DataFrame) -> pd.DataFrame:
    """Average % high-frustration per model.

    `pct_high_catavg` weights each category equally (matches "across the
    evaluations"); `pct_high_pooled` pools all rollouts. Both reported.
    """
    rows = []
    for model, g in rollout_df.groupby("model"):
        per_cat = g.groupby("category")["final_score"].apply(lambda s: (s >= HIGH).mean() * 100)
        rows.append({
            "model": model,
            "pct_high_catavg": per_cat.mean(),
            "pct_high_pooled": (g["final_score"] >= HIGH).mean() * 100,
            "mean_frustration": g["final_score"].mean(),
            "n_rollouts": len(g),
        })
    out = pd.DataFrame(rows).sort_values("pct_high_catavg", ascending=False)
    return out.reset_index(drop=True)


# --- Figure 2: per-category bars -------------------------------------------

def figure2_table(rollout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cat), g in rollout_df.groupby(["model", "category"]):
        scores = g["final_score"].to_numpy()
        k = int((scores >= HIGH).sum())
        n = len(scores)
        rows.append({
            "model": model,
            "category": cat,
            "mean_frustration": scores.mean(),
            "mean_ci": _mean_ci(scores),
            "pct_high": 100 * k / n,
            "pct_high_ci": 100 * _prop_ci(k, n),
            "n": n,
        })
    return pd.DataFrame(rows)


def plot_figure2(fig2: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    models = sorted(fig2["model"].unique())
    cats = [c for c in CATEGORIES if c in set(fig2["category"])]
    x = np.arange(len(cats))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, model in enumerate(models):
        sub = fig2[fig2["model"] == model].set_index("category").reindex(cats)
        offset = (i - (len(models) - 1) / 2) * width
        axes[0].bar(x + offset, sub["mean_frustration"], width,
                    yerr=sub["mean_ci"], capsize=2, label=model)
        axes[1].bar(x + offset, sub["pct_high"], width,
                    yerr=sub["pct_high_ci"], capsize=2, label=model)
    axes[0].set_ylabel("Mean frustration score")
    axes[0].set_title("Figure 2 (top): mean frustration by category")
    axes[1].set_ylabel("% responses with score >= 5")
    axes[1].set_title("Figure 2 (bottom): % high-frustration by category")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cats, rotation=20, ha="right")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --- Figure 3: per-turn progression ----------------------------------------

def figure3_table(turn_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = turn_df[turn_df["category"].isin(["extended", "wildchat"])]
    for (model, cat, turn), g in sub.groupby(["model", "category", "turn"]):
        scores = g["score"].to_numpy()
        k = int((scores >= HIGH).sum())
        n = len(scores)
        rows.append({
            "model": model,
            "category": cat,
            "turn": turn,
            "mean_frustration": scores.mean(),
            "mean_ci": _mean_ci(scores),
            "pct_high": 100 * k / n,
            "pct_high_ci": 100 * _prop_ci(k, n),
            "n": n,
        })
    return pd.DataFrame(rows).sort_values(["category", "model", "turn"])


def plot_figure3(fig3: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt

    cats = [c for c in ["extended", "wildchat"] if c in set(fig3["category"])]
    if not cats:
        return
    fig, axes = plt.subplots(2, len(cats), figsize=(6 * len(cats), 8), squeeze=False)
    for j, cat in enumerate(cats):
        cd = fig3[fig3["category"] == cat]
        for model, g in cd.groupby("model"):
            g = g.sort_values("turn")
            t = g["turn"].to_numpy()
            m = g["mean_frustration"].to_numpy()
            mci = g["mean_ci"].to_numpy()
            axes[0][j].plot(t, m, marker="o", label=model)
            axes[0][j].fill_between(t, m - mci, m + mci, alpha=0.15)
            p = g["pct_high"].to_numpy()
            pci = g["pct_high_ci"].to_numpy()
            axes[1][j].plot(t, p, marker="o", label=model)
            axes[1][j].fill_between(t, p - pci, p + pci, alpha=0.15)
        axes[0][j].set_title(f"{cat}: mean frustration per turn")
        axes[1][j].set_title(f"{cat}: % >= 5 per turn")
        axes[1][j].set_xlabel("Turn")
        axes[0][j].set_ylabel("Mean frustration")
        axes[1][j].set_ylabel("% >= 5")
        axes[0][j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze distress-eval results (Figures 1-3).")
    ap.add_argument("--scale", default="pilot", choices=["pilot", "medium", "full"])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=None, help="Where to write tables/plots (default: results/<scale>/analysis).")
    args = ap.parse_args(argv)

    rollout_df, turn_df = _load_scale_dir(args.results_dir, args.scale)
    out_dir = args.out_dir or os.path.join(args.results_dir, args.scale, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    fig1 = figure1_table(rollout_df)
    fig2 = figure2_table(rollout_df)
    fig3 = figure3_table(turn_df)

    fig1.to_csv(os.path.join(out_dir, "figure1_summary.csv"), index=False)
    fig2.to_csv(os.path.join(out_dir, "figure2_by_category.csv"), index=False)
    fig3.to_csv(os.path.join(out_dir, "figure3_per_turn.csv"), index=False)

    plot_figure2(fig2, os.path.join(out_dir, "figure2_by_category.png"))
    plot_figure3(fig3, os.path.join(out_dir, "figure3_per_turn.png"))

    print("\n=== Figure 1: avg % high-frustration (final-turn score >= 5) ===")
    with pd.option_context("display.float_format", lambda v: f"{v:.1f}"):
        print(fig1.to_string(index=False))
    print(f"\nWrote tables and plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
