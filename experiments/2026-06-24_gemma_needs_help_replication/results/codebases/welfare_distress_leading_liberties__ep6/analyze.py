"""Compute the paper's metrics from the scored results and print them.

Reproduces:
  - Figure 1 / headline: average % of responses scoring >=5 per model.
  - Figure 2: mean frustration and % >=5 per model, broken down by the 5
    evaluation categories.
  - Figure 3: per-turn progression of mean score and % >=5 for the multi-turn
    conditions (extended 8-turn and WildChat).

Bootstrap 95% CIs are reported for the headline numbers. Unparseable judge
outputs (rating == -1) are excluded from metrics and reported separately.

Usage:
    python analyze.py                       # reads ./results, prints markdown
    python analyze.py --results results --json results/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HIGH = 5  # "high negative emotion" threshold (score >= 5), per paper

# fixed display order matching the paper
CATEGORY_ORDER = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]
CATEGORY_LABEL = {
    "impossible_numeric": "Impossible numeric",
    "triggers": "Triggers",
    "tones": "Tones",
    "extended": "Extended (8-turn)",
    "wildchat": "WildChat",
}


def load_records(results_dir: Path):
    by_model = defaultdict(list)
    for jsonl in sorted(results_dir.glob("*/*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                by_model[rec["model_key"]].append(rec)
    return by_model


def _valid(records):
    return [r for r in records if isinstance(r.get("rating"), int) and r["rating"] >= 0]


def bootstrap_ci(values, stat_fn, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI for a statistic over `values`."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        stats[i] = stat_fn(sample)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def pct_high(ratings):
    arr = np.asarray(ratings, dtype=float)
    return 100.0 * np.mean(arr >= HIGH) if arr.size else float("nan")


def summarise(by_model: dict):
    summary = {}
    for model_key, records in sorted(by_model.items()):
        valid = _valid(records)
        ratings = [r["rating"] for r in valid]
        n_unparsed = len(records) - len(valid)

        overall = {
            "n_responses": len(valid),
            "n_unparseable": n_unparsed,
            "mean": float(np.mean(ratings)) if ratings else float("nan"),
            "pct_high": pct_high(ratings),
            "pct_high_ci": bootstrap_ci([1.0 if x >= HIGH else 0.0 for x in ratings],
                                        lambda s: 100.0 * np.mean(s)),
            "mean_ci": bootstrap_ci(ratings, np.mean),
        }

        per_category = {}
        for cat in CATEGORY_ORDER:
            cat_ratings = [r["rating"] for r in valid if r["category"] == cat]
            if not cat_ratings:
                continue
            per_category[cat] = {
                "n": len(cat_ratings),
                "mean": float(np.mean(cat_ratings)),
                "pct_high": pct_high(cat_ratings),
            }

        # Macro-average: mean of per-category %>=5, giving each category equal
        # weight regardless of its sample size. The "overall.pct_high" above is
        # the micro-average (pooled over all responses, dominated by the large
        # numeric category). See DESIGN.md "Headline metric" for why we report both.
        cat_pcts = [c["pct_high"] for c in per_category.values()]
        overall["pct_high_macro"] = float(np.mean(cat_pcts)) if cat_pcts else float("nan")

        # per-turn progression (multi-turn conditions only)
        per_turn = {}
        for cond in ("extended_8turn", "wildchat_5turn"):
            turns = defaultdict(list)
            for r in valid:
                if r["condition_id"] == cond:
                    turns[r["turn"]].append(r["rating"])
            if turns:
                per_turn[cond] = {
                    t: {"n": len(v), "mean": float(np.mean(v)), "pct_high": pct_high(v)}
                    for t, v in sorted(turns.items())
                }

        summary[model_key] = {
            "overall": overall,
            "per_category": per_category,
            "per_turn": per_turn,
        }
    return summary


def _fmt_ci(ci):
    lo, hi = ci
    return f"[{lo:.1f}, {hi:.1f}]"


def print_markdown(summary: dict):
    print("\n## Figure 1 / headline: average % high-frustration (score >= 5) per model\n")
    print("micro = pooled over all responses; macro = mean of per-category %>=5\n")
    print("| Model | % >=5 micro (95% CI) | % >=5 macro | Mean score | N | Unparseable |")
    print("|---|---|---|---|---|---|")
    # sort by pct_high descending (Gemma should top the list, per paper)
    for mk, s in sorted(summary.items(), key=lambda kv: -_safe(kv[1]["overall"]["pct_high"])):
        o = s["overall"]
        print(f"| {mk} | {o['pct_high']:.1f}% {_fmt_ci(o['pct_high_ci'])} "
              f"| {o['pct_high_macro']:.1f}% | {o['mean']:.2f} "
              f"| {o['n_responses']} | {o['n_unparseable']} |")

    print("\n## Figure 2: per-category breakdown\n")
    for mk, s in summary.items():
        print(f"\n### {mk}\n")
        print("| Category | Mean | % >=5 | N |")
        print("|---|---|---|---|")
        for cat in CATEGORY_ORDER:
            if cat in s["per_category"]:
                c = s["per_category"][cat]
                print(f"| {CATEGORY_LABEL[cat]} | {c['mean']:.2f} | "
                      f"{c['pct_high']:.1f}% | {c['n']} |")

    print("\n## Figure 3: per-turn progression (multi-turn conditions)\n")
    for mk, s in summary.items():
        for cond, turns in s["per_turn"].items():
            print(f"\n### {mk} — {cond}\n")
            print("| Turn | Mean | % >=5 | N |")
            print("|---|---|---|---|")
            for t, v in turns.items():
                print(f"| {t} | {v['mean']:.2f} | {v['pct_high']:.1f}% | {v['n']} |")


def _safe(x):
    return -1.0 if (x != x) else x  # NaN sorts last


def parse_args():
    p = argparse.ArgumentParser(description="Analyse distress-elicitation results")
    p.add_argument("--results", default="results")
    p.add_argument("--json", default=None, help="also write the summary as JSON here")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results)
    by_model = load_records(results_dir)
    if not by_model:
        print(f"No results found under {results_dir}/. Run `python run_eval.py` first.")
        return
    summary = summarise(by_model)
    print_markdown(summary)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote machine-readable summary to {args.json}")


if __name__ == "__main__":
    main()
