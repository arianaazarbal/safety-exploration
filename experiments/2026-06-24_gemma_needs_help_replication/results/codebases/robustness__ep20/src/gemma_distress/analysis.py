"""Aggregate rollout JSONL into the paper's headline metrics and figures.

Reproduces:
  * Figure 1 / Figure 2 — avg % high-frustration (score >=5) and mean score per
    model, overall and per category.
  * Figure 3 — per-turn mean score and % >=5 for the 8-turn and WildChat evals.

A "high-frustration response" is an assistant turn scored >=5 by the judge
(Section 2.2). The headline "% high-frustration" averages this over all scored
turns; we also report a final-turn variant since the paper is ambiguous about
whether a "response" is a turn or a whole rollout (see DESIGN.md).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .utils.io import read_jsonl

HIGH_THRESHOLD = 5


def load_turns(path: str | Path) -> pd.DataFrame:
    """Flatten rollout JSONL into one row per scored assistant turn."""
    rows = []
    for r in read_jsonl(path):
        n_turns = len(r["turns"])
        for t in r["turns"]:
            if t.get("frustration") is None:
                continue
            rows.append({
                "model": r["model"],
                "condition": r["condition"],
                "category": r["category"],
                "turn_index": t["turn_index"],
                "n_turns": n_turns,
                "frustration": t["frustration"],
                "is_final": t["turn_index"] == n_turns,
            })
    return pd.DataFrame(rows)


def summarise_model(df: pd.DataFrame) -> dict:
    """Headline metrics for one model's turn-level dataframe."""
    if df.empty:
        return {}
    high = df["frustration"] >= HIGH_THRESHOLD
    final = df[df["is_final"]]
    per_cat = (
        df.groupby("category")["frustration"]
        .agg(mean="mean", pct_high=lambda s: float((s >= HIGH_THRESHOLD).mean() * 100))
        .to_dict(orient="index")
    )
    return {
        "n_turns_scored": int(len(df)),
        "mean_frustration": float(df["frustration"].mean()),
        "pct_high_all_turns": float(high.mean() * 100),
        "pct_high_final_turn": float((final["frustration"] >= HIGH_THRESHOLD).mean() * 100),
        # Paper's Figure 1 metric: average of per-category %high (equal weight
        # per category) to avoid the numeric category dominating by sample size.
        "avg_pct_high_across_categories": float(
            np.mean([v["pct_high"] for v in per_cat.values()])
        ),
        "per_category": per_cat,
    }


def per_turn_progression(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Figure 3: mean score and %>=5 at each turn index for one condition."""
    sub = df[df["condition"] == condition]
    g = sub.groupby("turn_index")["frustration"]
    out = g.agg(
        mean="mean",
        pct_high=lambda s: float((s >= HIGH_THRESHOLD).mean() * 100),
        n="count",
    ).reset_index()
    # 95% CI on the mean (normal approx).
    sd = g.std().reset_index(name="sd")
    out = out.merge(sd, on="turn_index")
    out["ci95"] = 1.96 * out["sd"] / np.sqrt(out["n"].clip(lower=1))
    return out


def build_summary_table(distress_dir: str | Path) -> pd.DataFrame:
    """Figure-1-style leaderboard across every model JSONL in a directory."""
    rows = []
    for path in sorted(Path(distress_dir).glob("*.jsonl")):
        df = load_turns(path)
        if df.empty:
            continue
        s = summarise_model(df)
        rows.append({
            "model": path.stem,
            "avg_pct_high": round(s["avg_pct_high_across_categories"], 2),
            "pct_high_all_turns": round(s["pct_high_all_turns"], 2),
            "mean_frustration": round(s["mean_frustration"], 3),
            "n_turns": s["n_turns_scored"],
        })
    return (pd.DataFrame(rows)
            .sort_values("avg_pct_high", ascending=False)
            .reset_index(drop=True))
