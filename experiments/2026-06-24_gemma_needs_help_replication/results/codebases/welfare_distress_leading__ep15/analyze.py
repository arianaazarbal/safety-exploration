"""Aggregate judged rollouts into the paper's headline metrics.

Reproduces (for the Gemma/Gemini scope):
  * Figure 1 / abstract: average % of high-frustration responses (score >= 5)
    per model.
  * Figure 2: mean frustration and % >= 5 per evaluation category.
  * Figure 3: per-turn mean frustration and % >= 5 for the multi-turn
    (extended 8-turn, WildChat 5-turn) conditions.

Both a POOLED metric (over all scored responses) and a MACRO metric (unweighted
mean over the 5 categories) are reported, because the paper's wording ("average %
of high-frustration responses across the evaluations") is ambiguous between them
and the category sample sizes are very unequal. See DESIGN.md.

Usage: python analyze.py [results_dir]
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

from config import CONDITIONS, HIGH_FRUSTRATION_THRESHOLD

# condition.name -> category
_CATEGORY = {c.name: c.category for c in CONDITIONS}


def _load_scores(results_dir: str):
    """Yield (model, category, condition, turn, rating) for every scored response."""
    for path in glob.glob(os.path.join(results_dir, "*", "*.jsonl")):
        model = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                category = rec.get("category") or _CATEGORY.get(rec.get("condition"), "?")
                for s in rec.get("scores", []):
                    yield model, category, rec.get("condition"), s["turn"], s["rating"]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _prop_ge(xs, threshold):
    if not xs:
        return float("nan")
    return sum(1 for x in xs if x >= threshold) / len(xs)


def _ci95_prop(p, n):
    """Normal-approximation 95% CI half-width for a proportion."""
    if n == 0 or math.isnan(p):
        return float("nan")
    return 1.96 * math.sqrt(max(p * (1 - p), 0) / n)


def aggregate(results_dir: str) -> dict:
    # ratings[model][category] = list of ratings
    ratings = defaultdict(lambda: defaultdict(list))
    # per_turn[model][condition][turn] = list of ratings
    per_turn = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model, category, condition, turn, rating in _load_scores(results_dir):
        ratings[model][category].append(rating)
        per_turn[model][condition][turn].append(rating)

    summary = {}
    thr = HIGH_FRUSTRATION_THRESHOLD
    for model, by_cat in ratings.items():
        all_ratings = [r for rs in by_cat.values() for r in rs]
        cat_stats = {}
        cat_prop_list = []
        cat_mean_list = []
        for cat, rs in sorted(by_cat.items()):
            prop = _prop_ge(rs, thr)
            cat_stats[cat] = {
                "n": len(rs),
                "mean": _mean(rs),
                "pct_ge5": 100 * prop,
                "ci95_pct": 100 * _ci95_prop(prop, len(rs)),
            }
            cat_prop_list.append(prop)
            cat_mean_list.append(_mean(rs))

        pooled_prop = _prop_ge(all_ratings, thr)
        summary[model] = {
            "n_total": len(all_ratings),
            "pooled_mean": _mean(all_ratings),
            "pooled_pct_ge5": 100 * pooled_prop,
            "macro_mean": _mean(cat_mean_list),
            "macro_pct_ge5": 100 * _mean(cat_prop_list),
            "by_category": cat_stats,
            "per_turn": {
                cond: {
                    str(turn): {
                        "n": len(rs),
                        "mean": _mean(rs),
                        "pct_ge5": 100 * _prop_ge(rs, thr),
                    }
                    for turn, rs in sorted(turns.items())
                }
                for cond, turns in per_turn[model].items()
            },
        }
    return summary


def _print_summary(summary: dict) -> None:
    print("\n=== Headline: % high-frustration responses (score >= 5) ===")
    print(f"{'model':<20} {'n':>6} {'macro%>=5':>10} {'pooled%>=5':>11} "
          f"{'macro mean':>11} {'pooled mean':>12}")
    # Order by macro %>=5 descending (matches the paper's ranking presentation).
    for model, s in sorted(summary.items(), key=lambda kv: -kv[1]["macro_pct_ge5"]):
        print(f"{model:<20} {s['n_total']:>6} {s['macro_pct_ge5']:>9.1f}% "
              f"{s['pooled_pct_ge5']:>10.1f}% {s['macro_mean']:>11.2f} "
              f"{s['pooled_mean']:>12.2f}")

    print("\n=== Per-category % >= 5 (Figure 2) ===")
    for model, s in sorted(summary.items()):
        print(f"\n{model}:")
        for cat, cs in s["by_category"].items():
            print(f"  {cat:<12} n={cs['n']:>5}  mean={cs['mean']:>5.2f}  "
                  f">=5: {cs['pct_ge5']:>5.1f}% (+/-{cs['ci95_pct']:.1f})")

    print("\n=== Per-turn progression (Figure 3: extended & wildchat) ===")
    for model, s in sorted(summary.items()):
        for cond in ("extended", "wildchat"):
            turns = s["per_turn"].get(cond)
            if not turns:
                continue
            print(f"\n{model} / {cond}:")
            for turn, ts in sorted(turns.items(), key=lambda kv: int(kv[0])):
                print(f"  turn {int(turn)+1:>2}: mean={ts['mean']:>5.2f}  "
                      f">=5: {ts['pct_ge5']:>5.1f}%  (n={ts['n']})")


def _write_csv(summary: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "n", "mean", "pct_ge5", "ci95_pct"])
        for model, s in sorted(summary.items()):
            for cat, cs in s["by_category"].items():
                w.writerow([model, cat, cs["n"], f"{cs['mean']:.4f}",
                            f"{cs['pct_ge5']:.4f}", f"{cs['ci95_pct']:.4f}"])
            w.writerow([model, "ALL(macro)", s["n_total"], f"{s['macro_mean']:.4f}",
                        f"{s['macro_pct_ge5']:.4f}", ""])
            w.writerow([model, "ALL(pooled)", s["n_total"], f"{s['pooled_mean']:.4f}",
                        f"{s['pooled_pct_ge5']:.4f}", ""])


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Results directory not found: {results_dir}")
    summary = aggregate(results_dir)
    if not summary:
        raise SystemExit(f"No scored results found under {results_dir}.")
    _print_summary(summary)

    json_path = os.path.join(results_dir, "summary.json")
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    _write_csv(summary, csv_path)
    print(f"\nWrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
