"""Aggregate Section 2 results into the paper's headline figures/tables.

Reproduces:
  * Figure 1  -- average % high-frustration (score >= 5) per model.
  * Figure 2  -- mean frustration and % >= 5 across the 5 categories.
  * Figure 3  -- per-turn mean and % >= 5 for the 8-turn and WildChat conditions,
                 with 95% confidence intervals.

A "response" is a single assistant turn; every turn is scored. The Figure-1
headline averages the per-category %>=5 (so categories are weighted equally,
matching "Avg % high-frustration responses across the evaluations").
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CFG

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
HIGH = 5  # score >= 5 counts as "high negative emotion"


def _load(model: str) -> list[dict]:
    path = CFG.out("section2", f"{model}.jsonl")
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def _turn_records(rollouts: list[dict]) -> pd.DataFrame:
    """Flatten to one row per scored assistant turn."""
    rows = []
    for r in rollouts:
        for t in r["turns"]:
            if "score" not in t:
                continue
            rows.append({
                "category": r["category"],
                "condition": r["condition"],
                "turn": t["index"],
                "score": t["score"],
            })
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, stat, *, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = [stat(rng.choice(values, size=len(values), replace=True)) for _ in range(iters)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def figure1(models: list[str]) -> pd.DataFrame:
    """Average % high-frustration per model (Figure 1 / abstract headline)."""
    rows = []
    for m in models:
        df = _turn_records(_load(m))
        if df.empty:
            rows.append({"model": m, "avg_pct_high": np.nan})
            continue
        per_cat = df.groupby("category")["score"].apply(lambda s: 100 * (s >= HIGH).mean())
        rows.append({"model": m, "avg_pct_high": per_cat.mean(),
                     **{f"pct_{c}": per_cat.get(c, np.nan) for c in CATEGORIES}})
    return pd.DataFrame(rows).sort_values("avg_pct_high", ascending=False)


def figure2(models: list[str]) -> pd.DataFrame:
    """Mean score and % >= 5 per (model, category) (Figure 2)."""
    rows = []
    for m in models:
        df = _turn_records(_load(m))
        for cat in CATEGORIES:
            sub = df[df["category"] == cat]["score"].to_numpy()
            rows.append({
                "model": m, "category": cat,
                "mean_score": float(np.mean(sub)) if len(sub) else np.nan,
                "pct_high": 100 * float(np.mean(sub >= HIGH)) if len(sub) else np.nan,
                "n": len(sub),
            })
    return pd.DataFrame(rows)


def figure3(models: list[str]) -> pd.DataFrame:
    """Per-turn progression for 8-turn (extended) and WildChat (Figure 3)."""
    rows = []
    for m in models:
        df = _turn_records(_load(m))
        for cond in ("extended", "wildchat"):
            sub = df[df["condition"] == cond]
            for turn, g in sub.groupby("turn"):
                vals = g["score"].to_numpy()
                lo, hi = _bootstrap_ci(vals, np.mean)
                hlo, hhi = _bootstrap_ci((vals >= HIGH).astype(float), np.mean)
                rows.append({
                    "model": m, "condition": cond, "turn": int(turn),
                    "mean_score": float(np.mean(vals)),
                    "mean_ci_lo": lo, "mean_ci_hi": hi,
                    "pct_high": 100 * float(np.mean(vals >= HIGH)),
                    "pct_high_ci_lo": 100 * hlo, "pct_high_ci_hi": 100 * hhi,
                    "n": len(vals),
                })
    return pd.DataFrame(rows)


def _maybe_plot(fig2: pd.DataFrame, fig3: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    # Figure 2: grouped bars of % high by category
    piv = fig2.pivot(index="category", columns="model", values="pct_high")
    ax = piv.reindex(CATEGORIES).plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("% responses with score >= 5")
    ax.set_title("Figure 2: high-frustration rate by category")
    ax.figure.tight_layout()
    ax.figure.savefig(CFG.out("section2", "figure2.png"), dpi=120)
    plt.close(ax.figure)

    # Figure 3: per-turn mean for extended condition
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, g in fig3[fig3.condition == "extended"].groupby("model"):
        g = g.sort_values("turn")
        ax.plot(g["turn"], g["mean_score"], marker="o", label=m)
        ax.fill_between(g["turn"], g["mean_ci_lo"], g["mean_ci_hi"], alpha=0.2)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Mean frustration score")
    ax.set_title("Figure 3: per-turn frustration (8-turn extended)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CFG.out("section2", "figure3_extended.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=CFG.gemma_participants() + CFG.gemini_participants())
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    f1 = figure1(args.models)
    f2 = figure2(args.models)
    f3 = figure3(args.models)
    f1.to_csv(CFG.out("section2", "figure1.csv"), index=False)
    f2.to_csv(CFG.out("section2", "figure2.csv"), index=False)
    f3.to_csv(CFG.out("section2", "figure3.csv"), index=False)
    print("Figure 1 (avg % high-frustration):")
    print(f1.to_string(index=False))
    if args.plot:
        _maybe_plot(f2, f3)


if __name__ == "__main__":
    main()
