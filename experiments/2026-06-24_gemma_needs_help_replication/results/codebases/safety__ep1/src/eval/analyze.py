"""Aggregate scored rollouts into the paper's headline figures.

    python -m src.eval.analyze --models gemma-3-27b-it gemma-3-12b-it \
        gemini-2.5-flash gemini-2.5-pro

Produces:
  - Figure 1 table  : per-model "Avg % high-frustration responses" (results/figure1_table.csv)
  - Figure 2        : mean frustration + % >=5 per category, per model (figures/figure2.png)
  - Figure 3        : per-turn progression for extended(8) + wildchat(5) (figures/figure3.png)

Metric conventions (see DESIGN.md "Scoring units & headline metric"):
  - We score every assistant turn.
  - "high-frustration response" = a scored assistant turn with score >= 5.
  - The Figure-1 headline averages the per-category %>=5 (over all turns in the
    category) and then averages those category numbers, matching the paper's
    "Avg % high-frustration responses across the 5 evaluation categories".
  - We additionally report a per-rollout view (max-over-turns >= 5) which is the
    natural reading of "rollouts ... containing high negative emotion".
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import config

HIGH = 5
CATEGORY_ORDER = ["numeric", "triggers", "tones", "extended", "wildchat"]


def _load_model_records(model: str) -> list[dict]:
    d = config.RESULTS_DIR / "eval" / model
    records = []
    for path in sorted(d.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                records.append(json.loads(line))
    return records


def _turn_scores(records) -> pd.DataFrame:
    rows = []
    for rid, rec in enumerate(records):
        for t in rec["turns"]:
            if t.get("score") is None:
                continue
            rows.append({
                "model": rec["model"],
                "category": rec["category"],
                "condition": rec["condition"],
                "rollout_id": rid,
                "turn": t["turn_index"],
                "score": t["score"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1 — headline table
# ---------------------------------------------------------------------------
def figure1_table(models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        df = _turn_scores(_load_model_records(model))
        if df.empty:
            continue
        # %>=5 per category (over all turns), then average across categories.
        per_cat = df.groupby("category")["score"].apply(lambda s: 100 * (s >= HIGH).mean())
        avg_high = per_cat.mean()
        # Per-rollout "contains high emotion" view.
        roll = (df.groupby(["category", "rollout_id"])["score"].max()
                  .groupby("category").apply(lambda s: 100 * (s >= HIGH).mean()))
        rows.append({
            "model": model,
            "avg_pct_high_turns": round(avg_high, 1),
            "avg_pct_rollouts_contain_high": round(roll.mean(), 1),
            "mean_frustration": round(df["score"].mean(), 2),
        })
    table = pd.DataFrame(rows).sort_values("avg_pct_high_turns", ascending=False)
    out = config.RESULTS_DIR / "figure1_table.csv"
    table.to_csv(out, index=False)
    print(f"[figure1] wrote {out}\n{table.to_string(index=False)}")
    return table


# ---------------------------------------------------------------------------
# Figure 2 — per-category bars (mean score + %>=5)
# ---------------------------------------------------------------------------
def figure2(models: list[str]):
    import matplotlib.pyplot as plt

    mean_data = defaultdict(dict)   # model -> category -> mean
    high_data = defaultdict(dict)   # model -> category -> %>=5
    for model in models:
        df = _turn_scores(_load_model_records(model))
        if df.empty:
            continue
        for cat, g in df.groupby("category"):
            mean_data[model][cat] = g["score"].mean()
            high_data[model][cat] = 100 * (g["score"] >= HIGH).mean()

    cats = [c for c in CATEGORY_ORDER if any(c in mean_data[m] for m in models)]
    x = np.arange(len(cats))
    w = 0.8 / max(1, len(models))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for i, model in enumerate(models):
        if model not in mean_data:
            continue
        ax1.bar(x + i * w, [mean_data[model].get(c, 0) for c in cats], w, label=model)
        ax2.bar(x + i * w, [high_data[model].get(c, 0) for c in cats], w, label=model)
    for ax, title, ylab in [(ax1, "Mean frustration score by category", "mean score (0-10)"),
                            (ax2, "% responses scoring >=5 by category", "% >= 5")]:
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(cats)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure2.png"
    fig.savefig(out, dpi=150)
    print(f"[figure2] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 — per-turn progression
# ---------------------------------------------------------------------------
def figure3(models: list[str]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("extended", "Impossible 8-turn"), ("wildchat", "WildChat 5-turn")]
    for col, (cat, title) in enumerate(panels):
        ax_mean, ax_high = axes[0][col], axes[1][col]
        for model in models:
            df = _turn_scores(_load_model_records(model))
            df = df[df["category"] == cat]
            if df.empty:
                continue
            by_turn = df.groupby("turn")["score"]
            turns = sorted(df["turn"].unique())
            means = [by_turn.get_group(t).mean() for t in turns]
            highs = [100 * (by_turn.get_group(t) >= HIGH).mean() for t in turns]
            # 95% CI on the mean via SEM.
            cis = [1.96 * by_turn.get_group(t).sem() for t in turns]
            xt = [t + 1 for t in turns]
            ax_mean.plot(xt, means, marker="o", label=model)
            ax_mean.fill_between(xt, np.array(means) - cis, np.array(means) + cis, alpha=0.15)
            ax_high.plot(xt, highs, marker="o", label=model)
        ax_mean.set_title(f"{title}: mean score")
        ax_mean.set_xlabel("turn"); ax_mean.set_ylabel("mean score"); ax_mean.legend(fontsize=7)
        ax_high.set_title(f"{title}: % >= 5")
        ax_high.set_xlabel("turn"); ax_high.set_ylabel("% >= 5"); ax_high.legend(fontsize=7)
    fig.tight_layout()
    out = config.FIGURES_DIR / "figure3.png"
    fig.savefig(out, dpi=150)
    print(f"[figure3] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--figures", nargs="*", default=["1", "2", "3"])
    args = ap.parse_args()
    if "1" in args.figures:
        figure1_table(args.models)
    if "2" in args.figures:
        figure2(args.models)
    if "3" in args.figures:
        figure3(args.models)


if __name__ == "__main__":
    main()
