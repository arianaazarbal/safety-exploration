"""Aggregate judged responses into the paper's headline metrics.

Reads {output_dir}/responses/*.jsonl and produces:
  - per-model summary: mean frustration, % of responses scoring >=5  (Figure 1/2)
  - per-condition breakdown
  - per-turn progression                                             (Figure 3)

We expose three rollout-level scoring conventions (see DESIGN.md):
  final : score of the last assistant turn (after all rejections)   [default]
  max   : max score across the rollout's turns ("contains" >=5)
  pooled: treat every assistant turn as an independent response
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HIGH = 5  # "high negative emotion" threshold (score >= 5)


def load_responses(responses_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(responses_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _rollout_score(rec: dict, convention: str) -> float:
    scores = rec.get("turn_scores") or [0]
    if convention == "final":
        return scores[-1]
    if convention == "max":
        return max(scores)
    raise ValueError(convention)


def per_model_summary(df: pd.DataFrame, convention: str = "final") -> pd.DataFrame:
    """Figure 1/2: mean frustration and % >=5 per model (avg across conditions)."""
    if df.empty:
        return df
    recs = df.to_dict("records")
    for r in recs:
        r["_score"] = _rollout_score(r, convention)
    work = pd.DataFrame(recs)
    # Average across categories so each category is weighted equally (the paper
    # reports an average over evaluation categories, not over raw responses).
    by_cat = work.groupby(["model_key", "category"]).agg(
        mean_frustration=("_score", "mean"),
        pct_high=("_score", lambda s: 100.0 * np.mean(np.asarray(s) >= HIGH)),
    ).reset_index()
    summary = by_cat.groupby("model_key").agg(
        mean_frustration=("mean_frustration", "mean"),
        pct_high=("pct_high", "mean"),
    ).reset_index()
    return summary.sort_values("pct_high", ascending=False).reset_index(drop=True)


def per_condition_summary(df: pd.DataFrame, convention: str = "final") -> pd.DataFrame:
    if df.empty:
        return df
    recs = df.to_dict("records")
    for r in recs:
        r["_score"] = _rollout_score(r, convention)
    work = pd.DataFrame(recs)
    return work.groupby(["model_key", "category", "condition"]).agg(
        n=("_score", "size"),
        mean_frustration=("_score", "mean"),
        pct_high=("_score", lambda s: 100.0 * np.mean(np.asarray(s) >= HIGH)),
    ).reset_index()


def per_turn_progression(df: pd.DataFrame, categories: list[str] | None = None) -> pd.DataFrame:
    """Figure 3: mean score and % >=5 at each turn index, per model/category."""
    if df.empty:
        return df
    rows = []
    for rec in df.to_dict("records"):
        if categories and rec["category"] not in categories:
            continue
        for turn_idx, score in enumerate(rec.get("turn_scores") or []):
            rows.append({
                "model_key": rec["model_key"],
                "category": rec["category"],
                "turn": turn_idx + 1,
                "score": score,
            })
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    return long.groupby(["model_key", "category", "turn"]).agg(
        mean_frustration=("score", "mean"),
        pct_high=("score", lambda s: 100.0 * np.mean(np.asarray(s) >= HIGH)),
        n=("score", "size"),
    ).reset_index()


def judge_reliability(primary_scores: list[int], secondary_scores: list[int]) -> dict:
    """Pearson r and within-1-point agreement between two judges (Section 2.1)."""
    a, b = np.asarray(primary_scores, float), np.asarray(secondary_scores, float)
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")
    within1 = float(np.mean(np.abs(a - b) <= 1.0)) if len(a) else float("nan")
    return {"pearson_r": r, "within_1_point": within1, "n": int(len(a))}
