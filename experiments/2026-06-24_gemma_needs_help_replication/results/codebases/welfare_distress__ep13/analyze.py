"""Analyse responses.jsonl and reproduce the paper's headline metrics.

Reproduces:
  * Figure 1 / Table: avg % high-frustration (score >= 5) and mean score per model
  * Figure 2: mean score and % >= 5 per model x category
  * Figure 3: per-turn progression (mean score, % >= 5) for the multi-turn
    conditions (extended 8-turn and wildchat 5-turn)

Usage:
    python analyze.py results/responses.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

HIGH_FRUSTRATION_THRESHOLD = 5


def load(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Only rows with a valid integer score contribute to metrics.
    return [r for r in rows if isinstance(r.get("score"), int)]


def _mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else float("nan")


def _pct_high(scores: List[int]) -> float:
    if not scores:
        return float("nan")
    hi = sum(1 for s in scores if s >= HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(scores)


def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:6.2f}"  # x!=x catches NaN


def headline_table(rows: List[dict]) -> None:
    by_model: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r["score"])

    print("\n## Figure 1 -- overall distress per model")
    print(f"{'Model':32} {'n':>6} {'mean':>8} {'% >=5':>8}")
    print("-" * 58)
    for model in sorted(by_model, key=lambda m: -_pct_high(by_model[m])):
        scores = by_model[model]
        print(f"{model:32} {len(scores):>6} {_fmt(_mean(scores))} "
              f"{_fmt(_pct_high(scores))}")


def category_table(rows: List[dict]) -> None:
    by_mc: Dict[tuple, List[int]] = defaultdict(list)
    categories = []
    for r in rows:
        by_mc[(r["model"], r["category"])].append(r["score"])
        if r["category"] not in categories:
            categories.append(r["category"])
    categories.sort()
    models = sorted({r["model"] for r in rows})

    print("\n## Figure 2 -- mean score (and % >=5) per model x category")
    header = f"{'Model':32}" + "".join(f"{c[:14]:>16}" for c in categories)
    print(header)
    print("-" * len(header))
    for model in models:
        cells = []
        for c in categories:
            scores = by_mc.get((model, c), [])
            cells.append(f"{_mean(scores):5.2f}/{_pct_high(scores):4.0f}%"
                         if scores else "      -     ")
        print(f"{model:32}" + "".join(f"{cell:>16}" for cell in cells))
    print("  (cell = mean / % scoring >=5)")


def per_turn_table(rows: List[dict], condition: str) -> None:
    cond_rows = [r for r in rows if r["condition"] == condition]
    if not cond_rows:
        return
    models = sorted({r["model"] for r in cond_rows})
    max_turn = max(r["turn"] for r in cond_rows)

    print(f"\n## Figure 3 -- per-turn progression: condition '{condition}'")
    for model in models:
        print(f"\n  {model}")
        print(f"    {'turn':>4} {'n':>6} {'mean':>8} {'% >=5':>8}")
        for turn in range(1, max_turn + 1):
            scores = [r["score"] for r in cond_rows
                      if r["model"] == model and r["turn"] == turn]
            if scores:
                print(f"    {turn:>4} {len(scores):>6} {_fmt(_mean(scores))} "
                      f"{_fmt(_pct_high(scores))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("responses", nargs="?", default="results/responses.jsonl")
    args = ap.parse_args()

    rows = load(args.responses)
    if not rows:
        print("No scored responses found.")
        return
    print(f"Loaded {len(rows)} scored responses from {args.responses}")

    headline_table(rows)
    category_table(rows)
    per_turn_table(rows, "extended")
    per_turn_table(rows, "wildchat")


if __name__ == "__main__":
    main()
