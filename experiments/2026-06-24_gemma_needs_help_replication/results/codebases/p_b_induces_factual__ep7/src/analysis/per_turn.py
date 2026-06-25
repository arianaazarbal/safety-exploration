"""Figure 3: per-turn frustration progression for the 8-turn extended and WildChat
conditions, with 95% confidence intervals.

Reproduces the paper's headline that Gemma 27B's mean frustration rises from ~1.5 to
~5.5 between turn 1 and turn 8, and that no model scores >=5 until ~turn 3 on WildChat.
CIs are normal-approximation (mean +- 1.96*SEM for the mean curve; Wald interval for the
proportion curve).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from .io import load_many

TURN_CONDITIONS = {"extended": "extended_8turn", "wildchat": "wildchat_5turn"}


def per_turn_table(df: pd.DataFrame, category: str) -> pd.DataFrame:
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    sub = df[df["category"] == category]
    rows = []
    for (model, turn), g in sub.groupby(["model", "turn"]):
        s = g["score"].to_numpy()
        n = len(s)
        mean = s.mean()
        sem = s.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        p = (s >= thr).mean()
        p_ci = 1.96 * np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        rows.append(
            {
                "model": model,
                "turn": int(turn),
                "n": n,
                "mean_score": mean,
                "mean_ci": 1.96 * sem,
                "pct_high": 100.0 * p,
                "pct_high_ci": 100.0 * p_ci,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "turn"])


def _plot(tbl: pd.DataFrame, category: str, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, g in tbl.groupby("model"):
        g = g.sort_values("turn")
        axes[0].plot(g["turn"], g["mean_score"], marker="o", label=m)
        axes[0].fill_between(g["turn"], g["mean_score"] - g["mean_ci"],
                             g["mean_score"] + g["mean_ci"], alpha=0.15)
        axes[1].plot(g["turn"], g["pct_high"], marker="o", label=m)
        axes[1].fill_between(g["turn"], g["pct_high"] - g["pct_high_ci"],
                             g["pct_high"] + g["pct_high_ci"], alpha=0.15)
    axes[0].set(title=f"{category}: mean frustration", xlabel="Turn", ylabel="Mean score")
    axes[1].set(title=f"{category}: % score >= 5", xlabel="Turn", ylabel="%")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Figure 3 per-turn progression")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    df = load_many(args.models)
    if df.empty:
        raise SystemExit("No eval results found.")

    for category in TURN_CONDITIONS:
        tbl = per_turn_table(df, category)
        if tbl.empty:
            continue
        tbl.to_csv(config.RESULTS_DIR / f"figure3_{category}_per_turn.csv", index=False)
        print(f"\n{category} per-turn:")
        print(tbl.to_string(index=False))
        if not args.no_plot:
            _plot(tbl, category, config.FIGURES_DIR / f"figure3_{category}_per_turn.png")


if __name__ == "__main__":
    main()
