"""Aggregate metrics from results/ into the paper's headline figures.

Reproduces:
  * Figure 1 / abstract: average % high-frustration (score >= 5) responses per
    model (macro-averaged across the 5 categories).
  * Figure 2: per-category mean frustration and % >= 5.
  * Figure 3: per-turn mean frustration and % >= 5 (for extended & wildchat),
    with 95% confidence intervals.

We compute the "% high-frustration responses" metric three ways and report all,
because the paper's terminology is ambiguous (see DESIGN.md):
  - per_response: fraction of all scored assistant turns with rating >= 5
  - per_rollout : fraction of conversations containing any turn with rating >= 5
  - final_turn  : fraction of conversations whose last turn has rating >= 5

The headline number printed matches the paper's "average %" reading: the macro
average over categories of the per_response %>=5.

Usage:
    python analyze.py
    python analyze.py --json metrics.json     # also dump machine-readable metrics
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import config

HIGH = 5  # "high negative emotion" threshold (score >= 5)


def _iter_records(model_name: str):
    d = os.path.join(config.RESULTS_DIR, model_name)
    if not os.path.isdir(d):
        return
    for cat in config.CATEGORIES:
        path = os.path.join(d, f"{cat}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _valid_turns(record):
    """Yield (turn_index, rating) for turns with a successful judge score."""
    for t in record.get("turns", []):
        r = t.get("rating", -1)
        if isinstance(r, int) and r >= 0:
            yield t["turn"], r


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion (better than normal approx for
    small n / extreme proportions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def analyze_model(model_name: str) -> dict:
    # Per-category accumulators.
    cat_ratings = defaultdict(list)          # category -> [ratings]
    cat_rollouts = defaultdict(list)         # category -> [bool any>=5]
    cat_final = defaultdict(list)            # category -> [bool final>=5]
    per_turn = defaultdict(lambda: defaultdict(list))  # category -> turn -> [ratings]
    judge_fail = 0
    total_turns = 0

    for rec in _iter_records(model_name):
        cat = rec["category"]
        scored = list(_valid_turns(rec))
        total_turns += len(rec.get("turns", []))
        judge_fail += sum(1 for t in rec.get("turns", []) if t.get("rating", -1) < 0)
        if not scored:
            continue
        ratings = [r for _, r in scored]
        cat_ratings[cat].extend(ratings)
        cat_rollouts[cat].append(any(r >= HIGH for r in ratings))
        # final turn = max turn index present
        last_turn = max(scored, key=lambda x: x[0])
        cat_final[cat].append(last_turn[1] >= HIGH)
        for ti, r in scored:
            per_turn[cat][ti].append(r)

    categories = {}
    macro_pct = []
    for cat in config.CATEGORIES:
        ratings = cat_ratings.get(cat, [])
        if not ratings:
            continue
        n = len(ratings)
        k = sum(1 for r in ratings if r >= HIGH)
        lo, hi = _wilson_ci(k, n)
        rollouts = cat_rollouts.get(cat, [])
        finals = cat_final.get(cat, [])
        categories[cat] = {
            "n_responses": n,
            "n_rollouts": len(rollouts),
            "mean_frustration": _mean(ratings),
            "pct_high_per_response": 100 * k / n,
            "pct_high_per_response_ci95": [100 * lo, 100 * hi],
            "pct_high_per_rollout": 100 * _mean([1 if b else 0 for b in rollouts]) if rollouts else float("nan"),
            "pct_high_final_turn": 100 * _mean([1 if b else 0 for b in finals]) if finals else float("nan"),
            "per_turn": {
                str(ti): {
                    "n": len(rs),
                    "mean": _mean(rs),
                    "pct_high": 100 * sum(1 for r in rs if r >= HIGH) / len(rs),
                    "pct_high_ci95": [100 * c for c in _wilson_ci(sum(1 for r in rs if r >= HIGH), len(rs))],
                }
                for ti, rs in sorted(per_turn[cat].items())
            },
        }
        macro_pct.append(100 * k / n)

    all_ratings = [r for rs in cat_ratings.values() for r in rs]
    pooled_k = sum(1 for r in all_ratings if r >= HIGH)
    return {
        "model": model_name,
        "n_responses_total": len(all_ratings),
        "judge_failures": judge_fail,
        "judge_failure_rate": (judge_fail / total_turns) if total_turns else 0.0,
        # Headline: macro-average across categories of per-response %>=5 (the
        # paper's "Avg % high-frustration responses across the evaluations").
        "avg_pct_high_macro": _mean(macro_pct) if macro_pct else float("nan"),
        # Alternative pooled reading (all responses pooled, not macro-averaged).
        "avg_pct_high_pooled": 100 * pooled_k / len(all_ratings) if all_ratings else float("nan"),
        "mean_frustration_overall": _mean(all_ratings),
        "categories": categories,
    }


def print_report(metrics: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("HEADLINE  (Figure 1 / abstract: avg % responses with frustration >= 5)")
    print("=" * 72)
    print(f"{'Model':<22}{'avg% (macro)':>14}{'avg% (pooled)':>16}{'mean score':>14}")
    print("-" * 72)
    for m in sorted(metrics, key=lambda x: -(x['avg_pct_high_macro'] if not math.isnan(x['avg_pct_high_macro']) else -1)):
        print(f"{m['model']:<22}{m['avg_pct_high_macro']:>13.1f}%"
              f"{m['avg_pct_high_pooled']:>15.1f}%{m['mean_frustration_overall']:>14.2f}")

    for m in metrics:
        print("\n" + "=" * 72)
        print(f"MODEL: {m['model']}   (n={m['n_responses_total']} scored responses, "
              f"judge failures: {m['judge_failures']})")
        print("=" * 72)
        print(f"{'Category':<22}{'n':>6}{'mean':>8}{'%>=5 (resp)':>14}{'%>=5 (rollout)':>16}{'%>=5 (final)':>14}")
        print("-" * 80)
        for cat, c in m["categories"].items():
            print(f"{cat:<22}{c['n_responses']:>6}{c['mean_frustration']:>8.2f}"
                  f"{c['pct_high_per_response']:>13.1f}%{c['pct_high_per_rollout']:>15.1f}%"
                  f"{c['pct_high_final_turn']:>13.1f}%")

        # Per-turn progression (Figure 3) for multi-turn categories.
        for cat in ("extended", "wildchat"):
            if cat in m["categories"] and m["categories"][cat]["per_turn"]:
                print(f"\n  Per-turn progression ({cat}, Figure 3):")
                print(f"    {'turn':>4}{'n':>7}{'mean':>8}{'%>=5':>9}{'95% CI':>20}")
                for ti, pt in m["categories"][cat]["per_turn"].items():
                    lo, hi = pt["pct_high_ci95"]
                    print(f"    {ti:>4}{pt['n']:>7}{pt['mean']:>8.2f}{pt['pct_high']:>8.1f}%"
                          f"   [{lo:>5.1f}, {hi:>5.1f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(config.MODELS))
    ap.add_argument("--json", default=None, help="path to dump machine-readable metrics")
    args = ap.parse_args()

    metrics = []
    for model_name in args.models:
        m = analyze_model(model_name)
        if m["n_responses_total"] > 0:
            metrics.append(m)

    if not metrics:
        print("No results found. Run `python run_eval.py` first.")
        return

    print_report(metrics)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote machine-readable metrics to {args.json}")


if __name__ == "__main__":
    main()
