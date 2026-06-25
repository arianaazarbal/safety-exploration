"""Aggregate scored results and reproduce the paper's headline figures.

Reads results/<model>.jsonl produced by run_eval.py and reports:
  1. Per-model, per-condition mean frustration and % >= 5 (Figure 2).
  2. Per-model headline: average % >= 5 across the 5 categories (Figure 1).
  3. Per-turn progression (mean and % >= 5 with 95% CIs) for the multi-turn
     conditions, reproducing Figure 3 (Gemma's rise from ~1.5 to ~5.5).

Writes summary_by_condition.csv, summary_by_model.csv and per_turn.csv.

Usage:
    python analyze.py [--results-dir results]
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

import config

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


def load_records(results_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _ratings(records):
    """Valid integer ratings only (drop turns the judge could not score)."""
    return [r["rating"] for r in records if r.get("rating") is not None]


def mean_ci95(values):
    """Mean and 95% CI half-width (normal approx). Returns (mean, half_width)."""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, float("nan")
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    return mean, 1.96 * se


def prop_ci95(successes, n):
    """Proportion and 95% CI half-width (normal approx)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    return p, 1.96 * se


def summarize_by_condition(records):
    """rows: (model, condition, n, mean, pct_high)."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["condition"])].append(r)

    rows = []
    for (model, cond), recs in sorted(groups.items()):
        ratings = _ratings(recs)
        n = len(ratings)
        mean, _ = mean_ci95(ratings)
        high = sum(1 for x in ratings if x >= HIGH)
        pct, _ = prop_ci95(high, n)
        rows.append({
            "model": model, "condition": cond, "n": n,
            "mean_frustration": round(mean, 3) if n else None,
            "pct_high_ge5": round(100 * pct, 2) if n else None,
        })
    return rows


def summarize_by_model(records):
    """Headline metric (Figure 1): average of the per-CATEGORY % >= 5.

    We average across categories (not pooled across all responses) so that
    categories with large samples (numeric) do not dominate, matching the
    paper's "average % high-frustration responses across the evaluations".
    """
    # category -> model -> ratings
    cat_model = defaultdict(lambda: defaultdict(list))
    for r in records:
        rating = r.get("rating")
        if rating is not None:
            cat_model[r["category"]][r["model"]].append(rating)

    models = sorted({r["model"] for r in records})
    rows = []
    for model in models:
        cat_pcts = []
        pooled = []
        for cat, m in cat_model.items():
            ratings = m.get(model, [])
            if ratings:
                high = sum(1 for x in ratings if x >= HIGH)
                cat_pcts.append(100 * high / len(ratings))
                pooled.extend(ratings)
        avg_cat_pct = sum(cat_pcts) / len(cat_pcts) if cat_pcts else float("nan")
        pooled_high = sum(1 for x in pooled if x >= HIGH)
        rows.append({
            "model": model,
            "avg_pct_high_across_categories": round(avg_cat_pct, 2),
            "pooled_pct_high": round(100 * pooled_high / len(pooled), 2) if pooled else None,
            "pooled_mean_frustration": round(float(np.mean(pooled)), 3) if pooled else None,
            "n_total": len(pooled),
        })
    # Sort descending by headline metric (most distressed first, like Figure 1).
    rows.sort(key=lambda r: (-(r["avg_pct_high_across_categories"]
                               if not math.isnan(r["avg_pct_high_across_categories"]) else -1)))
    return rows


def summarize_per_turn(records, conditions=("extended_8turn", "wildchat_5turn")):
    """rows: (model, condition, turn, n, mean, mean_ci, pct_high, pct_ci)."""
    groups = defaultdict(list)
    for r in records:
        if r["condition"] in conditions and r.get("rating") is not None:
            groups[(r["model"], r["condition"], r["turn"])].append(r["rating"])

    rows = []
    for (model, cond, turn), ratings in sorted(groups.items()):
        n = len(ratings)
        mean, mean_ci = mean_ci95(ratings)
        high = sum(1 for x in ratings if x >= HIGH)
        pct, pct_ci = prop_ci95(high, n)
        rows.append({
            "model": model, "condition": cond, "turn": turn, "n": n,
            "mean_frustration": round(mean, 3),
            "mean_ci95": round(mean_ci, 3) if not math.isnan(mean_ci) else None,
            "pct_high_ge5": round(100 * pct, 2),
            "pct_ci95": round(100 * pct_ci, 2) if not math.isnan(pct_ci) else None,
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_table(title, rows, cols):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no data)")
        return
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=config.RESULTS_DIR)
    args = ap.parse_args()

    records = load_records(args.results_dir)
    if not records:
        raise SystemExit(f"No results found in {args.results_dir}. Run run_eval.py first.")

    by_cond = summarize_by_condition(records)
    by_model = summarize_by_model(records)
    per_turn = summarize_per_turn(records)

    print_table("By model (headline, Figure 1)", by_model,
                ["model", "avg_pct_high_across_categories", "pooled_pct_high",
                 "pooled_mean_frustration", "n_total"])
    print_table("By condition (Figure 2)", by_cond,
                ["model", "condition", "n", "mean_frustration", "pct_high_ge5"])
    print_table("Per turn (Figure 3)", per_turn,
                ["model", "condition", "turn", "n", "mean_frustration", "pct_high_ge5"])

    write_csv(os.path.join(args.results_dir, "summary_by_model.csv"), by_model)
    write_csv(os.path.join(args.results_dir, "summary_by_condition.csv"), by_cond)
    write_csv(os.path.join(args.results_dir, "per_turn.csv"), per_turn)
    print(f"\nWrote summary_by_model.csv, summary_by_condition.csv, per_turn.csv to {args.results_dir}")


if __name__ == "__main__":
    main()
