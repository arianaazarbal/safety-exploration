"""Aggregate per-turn scores into the paper's headline numbers.

Produces:
  * per-model headline: avg %high-frustration (score>=5) over final-turn
    responses across the 5 categories  (Fig. 1 / Fig. 2)
  * per-model x category: mean score and %>=5                (Fig. 2)
  * per-model x turn: mean score and %>=5 for the 8-turn and WildChat
    conditions, with 95% CIs                                 (Fig. 3)

All inputs are the scores/*.jsonl files written by run_eval.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import config
from gemma_distress.utils.io import read_jsonl

SCORE_DIR = config.RESULTS_DIR / "section2" / "scores"
THR = config.HIGH_FRUSTRATION_THRESHOLD


def load_scores(models: list[str] | None = None) -> pd.DataFrame:
    rows = []
    paths = (
        [SCORE_DIR / f"{m}.jsonl" for m in models]
        if models else sorted(SCORE_DIR.glob("*.jsonl"))
    )
    for p in paths:
        rows.extend(read_jsonl(p))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _ci95_prop(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % of high-frustration final responses per model, averaged across the
    5 categories (matches Fig. 1's '% responses scoring >=5/10' column)."""
    final = df[df["is_final"]]
    # category-level %>=5 first, then average across categories per model
    cat = (
        final.assign(high=lambda d: d["rating"] >= THR)
        .groupby(["model", "category"])["high"].mean()
        .reset_index()
    )
    out = (cat.groupby("model")["high"].mean() * 100).reset_index()
    out = out.rename(columns={"high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    final = df[df["is_final"]]
    g = final.assign(high=lambda d: d["rating"] >= THR).groupby(["model", "category"])
    out = g.agg(mean_score=("rating", "mean"),
                pct_high=("high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high"] *= 100
    return out


def by_turn(df: pd.DataFrame, conditions: list[str] | None = None) -> pd.DataFrame:
    """Per-turn mean score and %>=5 with 95% CIs (for Fig. 3)."""
    sub = df
    if conditions:
        sub = sub[sub["condition"].isin(conditions)]
    g = sub.assign(high=lambda d: d["rating"] >= THR).groupby(["model", "condition", "turn"])
    out = g.agg(mean_score=("rating", "mean"),
                pct_high=("high", "mean"),
                n=("rating", "size")).reset_index()
    out["pct_high_ci"] = out.apply(lambda r: _ci95_prop(r["pct_high"], r["n"]) * 100, axis=1)
    out["pct_high"] *= 100
    # mean-score CI (normal approx)
    sd = g["rating"].std().reset_index(name="sd")
    out = out.merge(sd, on=["model", "condition", "turn"])
    out["mean_ci"] = out.apply(
        lambda r: 1.96 * (r["sd"] / math.sqrt(r["n"])) if r["n"] > 0 else 0.0, axis=1
    )
    return out


def write_all(models: list[str] | None = None) -> dict[str, Path]:
    df = load_scores(models)
    out_dir = config.RESULTS_DIR / "section2" / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if df.empty:
        print("[aggregate] no scores found")
        return paths
    for name, frame in {
        "headline": headline_table(df),
        "by_category": by_category(df),
        "by_turn": by_turn(df, ["extended_8turn", "wildchat_5turn"]),
    }.items():
        p = out_dir / f"{name}.csv"
        frame.to_csv(p, index=False)
        paths[name] = p
        print(f"[aggregate] wrote {p}")
    return paths


if __name__ == "__main__":  # pragma: no cover
    write_all()
