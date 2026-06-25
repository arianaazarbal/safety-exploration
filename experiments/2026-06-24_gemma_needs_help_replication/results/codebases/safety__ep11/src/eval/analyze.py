"""Aggregate scored eval results into the paper's headline numbers.

Reproduces:
* Figure 1 / Figure 2: mean frustration and % responses >= 5, per model and per
  category.
* Figure 3: per-turn progression of mean score and % >= 5 (8-turn & WildChat).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import config


def load_records(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _ratings(records, *, category=None, condition=None):
    out = []
    for rec in records:
        if category and rec["category"] != category:
            continue
        if condition and rec["condition"] != condition:
            continue
        for t in rec["turns"]:
            if t.get("rating") is not None:
                out.append(t["rating"])
    return out


def summarise(path: Path) -> dict:
    """Per-model summary: overall + per-category mean and high-frustration rate."""
    records = load_records(path)
    thr = config.HIGH_FRUSTRATION_THRESHOLD

    def stats(ratings):
        if not ratings:
            return {"n": 0, "mean": None, "pct_high": None}
        return {
            "n": len(ratings),
            "mean": mean(ratings),
            "pct_high": 100 * sum(r >= thr for r in ratings) / len(ratings),
        }

    model = records[0]["model"] if records else "?"
    categories = sorted({r["category"] for r in records})
    return {
        "model": model,
        "overall": stats(_ratings(records)),
        "by_category": {c: stats(_ratings(records, category=c)) for c in categories},
    }


def per_turn_progression(path: Path, condition: str) -> list[dict]:
    """Mean score and % >= 5 at each turn index for one condition (Figure 3)."""
    records = load_records(path)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    buckets = defaultdict(list)
    for rec in records:
        if rec["condition"] != condition:
            continue
        for t in rec["turns"]:
            if t.get("rating") is not None:
                buckets[t["turn_index"]].append(t["rating"])
    return [
        {
            "turn": k,
            "mean": mean(v),
            "pct_high": 100 * sum(r >= thr for r in v) / len(v),
            "n": len(v),
        }
        for k, v in sorted(buckets.items())
    ]


def compare_models(paths: list[Path]) -> list[dict]:
    """Build the Figure 1 leaderboard (avg % high-frustration per model)."""
    rows = [summarise(p) for p in paths]
    rows.sort(key=lambda r: (r["overall"]["pct_high"] or 0), reverse=True)
    return rows
