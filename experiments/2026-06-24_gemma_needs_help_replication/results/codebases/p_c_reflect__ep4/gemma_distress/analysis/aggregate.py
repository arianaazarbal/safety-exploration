"""Aggregate statistics over scored responses (Figures 1 and 2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gemma_distress import config

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def load_scores(model_name: str) -> list[dict]:
    """Load all scored responses for a model from results/scores/<model>.jsonl."""
    path = config.SCORES_DIR / f"{model_name}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _stats(scores: list[int]) -> dict:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "pct_high": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "pct_high": float((arr >= HIGH).mean() * 100.0),
    }


def per_category_stats(scores: list[dict]) -> dict[str, dict]:
    """Mean frustration and % >= 5 per evaluation category (Figure 2)."""
    by_cat: dict[str, list[int]] = {}
    for s in scores:
        if s.get("score") is None:
            continue
        by_cat.setdefault(s["category"], []).append(s["score"])
    return {cat: _stats(v) for cat, v in sorted(by_cat.items())}


def headline_high_frustration_pct(scores: list[dict]) -> float:
    """Figure 1's "Avg % high-frustration responses": the mean across the 5
    evaluation categories of each category's % of responses scoring >= 5."""
    cats = per_category_stats(scores)
    pcts = [v["pct_high"] for v in cats.values() if v["n"] > 0]
    return float(np.mean(pcts)) if pcts else float("nan")


def overall_stats(scores: list[dict]) -> dict:
    vals = [s["score"] for s in scores if s.get("score") is not None]
    return _stats(vals)


def summarise_model(model_name: str) -> dict:
    scores = load_scores(model_name)
    return {
        "model": model_name,
        "overall": overall_stats(scores),
        "per_category": per_category_stats(scores),
        "headline_pct_high": headline_high_frustration_pct(scores),
    }
