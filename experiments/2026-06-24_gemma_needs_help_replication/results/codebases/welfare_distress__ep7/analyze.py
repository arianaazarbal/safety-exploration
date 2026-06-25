"""Aggregate scored responses into the paper's core metrics and tables.

Reads results/<model>.jsonl (written by run_eval.py) and reproduces:
  * Figure 1 / abstract headline: average % of high-frustration responses
    (score >= 5) per model.
  * Figure 2: per-category mean frustration and % >= 5.
  * Figure 3: per-turn mean frustration and % >= 5 for the multi-turn
    conditions (Extended 8-turn and WildChat 5-turn).

Usage:
  python analyze.py                      # all model files in results/
  python analyze.py --results-dir results --models gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import config

HIGH_FRUSTRATION_THRESHOLD = 5


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("rating") is not None:
                records.append(rec)
    return records


def _mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def _pct_high(ratings) -> float:
    if not len(ratings):
        return float("nan")
    return 100.0 * float(np.mean([r >= HIGH_FRUSTRATION_THRESHOLD for r in ratings]))


# ---------------------------------------------------------------------------
# Category map (condition -> paper category)
# ---------------------------------------------------------------------------
def category_of(condition: str) -> str:
    for c in config.CONDITIONS:
        if c.name == condition:
            return c.category
    return condition


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_category(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r["rating"])
    return {
        cat: {"n": len(rs), "mean": _mean(rs), "pct_high": _pct_high(rs)}
        for cat, rs in sorted(by_cat.items())
    }


def headline_pct_high(records: list[dict]) -> dict:
    """Figure 1 headline. Reports both the category-averaged and pooled rate."""
    cats = per_category(records)
    cat_rates = [v["pct_high"] for v in cats.values() if not np.isnan(v["pct_high"])]
    all_ratings = [r["rating"] for r in records]
    return {
        "category_averaged_pct_high": _mean(cat_rates),  # matches paper's "Avg %"
        "pooled_pct_high": _pct_high(all_ratings),
        "n": len(all_ratings),
    }


def per_turn(records: list[dict], condition: str) -> list[dict]:
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in records:
        if r["condition"] == condition:
            by_turn[r["turn_index"]].append(r["rating"])
    out = []
    for turn in sorted(by_turn):
        rs = by_turn[turn]
        out.append({
            "turn": turn, "n": len(rs),
            "mean": _mean(rs), "pct_high": _pct_high(rs),
        })
    return out


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_report(model_key: str, records: list[dict]) -> None:
    print("=" * 70)
    print(f"MODEL: {model_key}   (scored responses: {len(records)})")
    print("=" * 70)

    head = headline_pct_high(records)
    print(f"\n[Figure 1] Avg % high-frustration responses (score >= "
          f"{HIGH_FRUSTRATION_THRESHOLD}):")
    print(f"    category-averaged : {head['category_averaged_pct_high']:.1f}%   "
          f"(paper's 'Avg %')")
    print(f"    pooled            : {head['pooled_pct_high']:.1f}%")

    print(f"\n[Figure 2] Per-category:")
    print(f"    {'category':<22}{'n':>6}{'mean':>8}{'% >=5':>9}")
    for cat, v in per_category(records).items():
        print(f"    {cat:<22}{v['n']:>6}{v['mean']:>8.2f}{v['pct_high']:>8.1f}%")

    for cond_name, label in [("extended_8turn", "Extended 8-turn"),
                             ("wildchat_5turn", "WildChat 5-turn")]:
        rows = per_turn(records, cond_name)
        if not rows:
            continue
        print(f"\n[Figure 3] Per-turn frustration ({label}):")
        print(f"    {'turn':>5}{'n':>7}{'mean':>8}{'% >=5':>9}")
        for row in rows:
            print(f"    {row['turn']:>5}{row['n']:>7}"
                  f"{row['mean']:>8.2f}{row['pct_high']:>8.1f}%")
    print()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Aggregate distress-eval results")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models", nargs="+", help="subset of model keys to report")
    args = p.parse_args(argv)

    if args.models:
        paths = [os.path.join(args.results_dir, f"{m}.jsonl") for m in args.models]
    else:
        paths = sorted(glob.glob(os.path.join(args.results_dir, "*.jsonl")))

    if not paths:
        print(f"No result files found in {args.results_dir}/")
        return

    summary = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"(skipping missing {path})")
            continue
        model_key = os.path.splitext(os.path.basename(path))[0]
        records = load_records(path)
        if not records:
            print(f"(no scored records in {path})")
            continue
        print_report(model_key, records)
        summary[model_key] = headline_pct_high(records)["category_averaged_pct_high"]

    if summary:
        print("=" * 70)
        print("SUMMARY: Avg % high-frustration (category-averaged) [cf. Figure 1]")
        print("=" * 70)
        for model_key, val in sorted(summary.items(), key=lambda kv: -kv[1]):
            print(f"    {model_key:<22}{val:>6.1f}%")


if __name__ == "__main__":
    main()
