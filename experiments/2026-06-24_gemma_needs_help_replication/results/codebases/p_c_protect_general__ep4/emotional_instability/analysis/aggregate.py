"""Aggregate Section 2 results into the Figure 1 / Figure 2 statistics.

Figure 1 (left): per-model "Avg % high-frustration responses" = the percentage
of responses scoring >= 5, averaged across the 5 evaluation categories (the
paper averages across categories, not across raw responses, so each category is
weighted equally).

Figure 2: per-model, per-category mean frustration and % >= 5.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np

from ..config import RESULTS_DIR

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_category_stats(path: str) -> dict:
    """{category: {mean, pct_ge5, n}} for one model's results file.

    A "response" is a single scored assistant turn (the paper scores each
    response on the 0–10 scale)."""
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in _load(path):
        cat = r.get("metadata", {}).get("category", r["condition"])
        for t in r["turns"]:
            if t["score"] is not None:
                by_cat[cat].append(int(t["score"]))
    out = {}
    for cat, scores in by_cat.items():
        arr = np.asarray(scores, dtype=float)
        out[cat] = {
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "pct_ge5": float((arr >= 5).mean() * 100) if len(arr) else float("nan"),
            "n": int(len(arr)),
        }
    return out


def figure1_table(results_glob: Optional[str] = None) -> list[dict]:
    """Per-model average %>=5 across categories (Figure 1 left). Sorted desc."""
    results_glob = results_glob or os.path.join(RESULTS_DIR, "section2", "*.jsonl")
    rows = []
    for path in glob.glob(results_glob):
        model = os.path.splitext(os.path.basename(path))[0]
        cats = per_category_stats(path)
        pcts = [cats[c]["pct_ge5"] for c in cats if not np.isnan(cats[c]["pct_ge5"])]
        avg = float(np.mean(pcts)) if pcts else float("nan")
        rows.append({"model": model, "avg_pct_high_frustration": avg,
                     "categories": cats})
    rows.sort(key=lambda r: (np.nan_to_num(r["avg_pct_high_frustration"], nan=-1)),
              reverse=True)
    return rows


def figure2_table(results_glob: Optional[str] = None) -> dict:
    """{model: {category: {mean, pct_ge5, n}}}."""
    results_glob = results_glob or os.path.join(RESULTS_DIR, "section2", "*.jsonl")
    out = {}
    for path in glob.glob(results_glob):
        model = os.path.splitext(os.path.basename(path))[0]
        out[model] = per_category_stats(path)
    return out


def write_summary(results_glob: Optional[str] = None, out_path: Optional[str] = None) -> dict:
    summary = {
        "figure1": figure1_table(results_glob),
        "figure2": figure2_table(results_glob),
    }
    out_path = out_path or os.path.join(RESULTS_DIR, "section2_summary.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
