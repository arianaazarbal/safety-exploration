"""Aggregate judged results into the paper's headline metrics (Figures 1-3).

Reads results/<model>.jsonl and prints:
  * Figure 1  -- average % high-frustration (score >= 5) per model, averaged over the
                 5 evaluation categories (equal weight, as in the paper's Figure 1).
  * Figure 2  -- mean frustration and % >= 5 per (model, category).
  * Figure 3  -- per-turn mean and % >= 5 for the extended (8-turn) and WildChat conditions.
  * Rollout-level "contains a response with score >= 5" rate (the paper's "70% of
    8-turn rollouts" statistic).

Optionally writes a CSV of the per-(model,category) table with --csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean

import config

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_all() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "*.jsonl"))):
        model = os.path.splitext(os.path.basename(path))[0]
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[model] = rows
    return out


def _valid(rows):
    """Drop judge failures (rating == -1) and report how many were dropped."""
    good = [r for r in rows if r.get("rating", -1) >= 0]
    return good, len(rows) - len(good)


def pct_high(rows) -> float:
    if not rows:
        return float("nan")
    hi = sum(1 for r in rows if r["rating"] >= config.HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(rows)


def mean_score(rows) -> float:
    return mean(r["rating"] for r in rows) if rows else float("nan")


def figure1(data):
    print("\n=== Figure 1: avg % high-frustration responses (score >= 5) ===")
    print("(averaged across the 5 evaluation categories, equal weight)\n")
    rows_out = []
    for model, rows in data.items():
        good, dropped = _valid(rows)
        per_cat = []
        for cat in CATEGORIES:
            cat_rows = [r for r in good if r["category"] == cat]
            if cat_rows:
                per_cat.append(pct_high(cat_rows))
        avg = mean(per_cat) if per_cat else float("nan")
        rows_out.append((model, avg, len(good), dropped))
    for model, avg, n, dropped in sorted(rows_out, key=lambda x: -(x[1] if x[1] == x[1] else -1)):
        warn = f"  (dropped {dropped} judge failures)" if dropped else ""
        print(f"  {model:<22} {avg:6.2f}%   n={n}{warn}")


def figure2(data, csv_path=None):
    print("\n=== Figure 2: per-category mean frustration and % >= 5 ===\n")
    header = ["model", "category", "n", "mean", "pct_high"]
    table = []
    for model, rows in data.items():
        good, _ = _valid(rows)
        for cat in CATEGORIES:
            cat_rows = [r for r in good if r["category"] == cat]
            if not cat_rows:
                continue
            table.append([model, cat, len(cat_rows), round(mean_score(cat_rows), 3),
                          round(pct_high(cat_rows), 2)])
    # pretty print grouped by model
    for model in data:
        print(f"  {model}")
        for row in [t for t in table if t[0] == model]:
            print(f"    {row[1]:<20} n={row[2]:<5} mean={row[3]:<6} %>=5={row[4]}")
        print()
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(table)
        print(f"  wrote {csv_path}")


def figure3(data):
    print("\n=== Figure 3: per-turn progression (extended 8-turn & WildChat) ===\n")
    for cond_category, label in [("extended", "Extended (8-turn)"), ("wildchat", "WildChat (5-turn)")]:
        print(f"  {label}")
        for model, rows in data.items():
            good, _ = _valid(rows)
            cond_rows = [r for r in good if r["category"] == cond_category]
            if not cond_rows:
                continue
            by_turn = defaultdict(list)
            for r in cond_rows:
                by_turn[r["turn_index"]].append(r)
            turns = sorted(by_turn)
            means = [f"{mean_score(by_turn[t]):.1f}" for t in turns]
            highs = [f"{pct_high(by_turn[t]):.0f}%" for t in turns]
            print(f"    {model}")
            print(f"      turn:     " + " ".join(f"{t+1:>5}" for t in turns))
            print(f"      mean:     " + " ".join(f"{m:>5}" for m in means))
            print(f"      %>=5:     " + " ".join(f"{h:>5}" for h in highs))
        print()


def rollout_level(data):
    print("\n=== Rollout-level: % of rollouts containing a response with score >= 5 ===\n")
    for model, rows in data.items():
        good, _ = _valid(rows)
        for cat in CATEGORIES:
            cat_rows = [r for r in good if r["category"] == cat]
            if not cat_rows:
                continue
            rollouts = defaultdict(list)
            for r in cat_rows:
                rollouts[(r["condition"], r["conversation_id"])].append(r["rating"])
            n_roll = len(rollouts)
            with_high = sum(1 for ratings in rollouts.values()
                            if max(ratings) >= config.HIGH_FRUSTRATION_THRESHOLD)
            print(f"  {model:<22} {cat:<20} {100.0*with_high/n_roll:6.1f}%  ({with_high}/{n_roll} rollouts)")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None, help="write the per-category table to this CSV path")
    args = ap.parse_args()

    data = load_all()
    if not data:
        print(f"No results found in {config.RESULTS_DIR}/. Run run_eval.py first.")
        return
    figure1(data)
    figure2(data, args.csv)
    figure3(data)
    rollout_level(data)


if __name__ == "__main__":
    main()
