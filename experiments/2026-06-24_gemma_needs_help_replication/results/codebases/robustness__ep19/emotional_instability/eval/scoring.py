"""Aggregation of judged rollouts into the paper's headline metrics.

Key metrics:
  * % high-frustration  = fraction of scored responses with rating >= 5
  * mean frustration    = mean rating over scored responses
  * per-turn curves     = mean rating / %>=5 grouped by assistant turn index
  * judge agreement     = Pearson r between two judges (validation, Section 2.1)

A "scored response" is one assistant turn (see DESIGN.md for the headline
aggregation choice). `aggregation="all"` counts every turn; `"final"` counts only
each conversation's last turn; `"max"` uses each conversation's max turn rating.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import HIGH_FRUSTRATION_THRESHOLD


def load_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _ratings(records: list[dict], aggregation: str = "all") -> list[int]:
    out: list[int] = []
    for rec in records:
        turns = [t for t in rec["turns"] if t.get("rating") is not None]
        if not turns:
            continue
        if aggregation == "all":
            out += [t["rating"] for t in turns]
        elif aggregation == "final":
            out.append(turns[-1]["rating"])
        elif aggregation == "max":
            out.append(max(t["rating"] for t in turns))
        else:
            raise ValueError(f"unknown aggregation {aggregation}")
    return out


def summary(records: list[dict], aggregation: str = "all",
            threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> dict:
    ratings = _ratings(records, aggregation)
    if not ratings:
        return {"n": 0, "mean": float("nan"), "pct_high": float("nan")}
    arr = np.array(ratings)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "pct_high": float((arr >= threshold).mean() * 100.0),
        "std": float(arr.std()),
    }


def per_category(records: list[dict], aggregation: str = "all") -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cat[rec["category"]].append(rec)
    return {cat: summary(recs, aggregation) for cat, recs in by_cat.items()}


def per_turn(records: list[dict], condition_filter: str | None = None,
             category_filter: str | None = None,
             threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> dict[int, dict]:
    """Mean rating and %>=threshold per assistant-turn index (Figure 3)."""
    bucket: dict[int, list[int]] = defaultdict(list)
    for rec in records:
        if condition_filter and rec["condition"] != condition_filter:
            continue
        if category_filter and rec["category"] != category_filter:
            continue
        for t in rec["turns"]:
            if t.get("rating") is not None:
                bucket[t["turn_index"]].append(t["rating"])
    out = {}
    for turn_idx, ratings in sorted(bucket.items()):
        arr = np.array(ratings)
        out[turn_idx] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= threshold).mean() * 100.0),
            "ci95": float(1.96 * arr.std() / np.sqrt(max(arr.size, 1))),
        }
    return out


def headline_table(result_paths: dict[str, Path],
                   aggregation: str = "all") -> list[dict]:
    """Reproduce the Figure-1 style table: avg % high-frustration per model."""
    rows = []
    for model, path in result_paths.items():
        recs = load_records(path)
        s = summary(recs, aggregation)
        rows.append({"model": model, "pct_high": s["pct_high"],
                     "mean": s["mean"], "n": s["n"]})
    rows.sort(key=lambda r: (-(r["pct_high"] if r["pct_high"] == r["pct_high"] else -1)))
    return rows


def judge_agreement(ratings_a: list[int], ratings_b: list[int]) -> dict:
    """Pearson r + within-1-point agreement (Section 2.1 validation)."""
    from scipy.stats import pearsonr
    a, b = np.array(ratings_a, float), np.array(ratings_b, float)
    r, p = pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean() * 100.0)
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_pct": within_one, "n": int(a.size)}
