"""Summarise eval results into the paper's headline metrics.

Reproduces, for the Gemma/Gemini subset:
  * Figure 1: average % high-frustration (score >= 5) responses per model.
  * Figure 2: mean frustration and % >= 5 per evaluation category.
  * Figure 3: per-turn progression of mean score and % >= 5 for the 8-turn
    (Extended) and WildChat conditions.

Reads every *.jsonl in the results dir (one scored response per line; produced
by run_eval.py), prints tables, and writes summary.json + summary CSVs.

We treat each scored assistant turn as one "response". The headline Figure-1
number is the mean of the 5 per-category % >= 5 rates (category-balanced, to
match the paper averaging "across the 5 evaluation categories"); we also report
the pooled rate over all responses. See DESIGN.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from statistics import mean

HIGH_THRESHOLD = 5  # "high negative emotion" = score >= 5 (Section 2.2)

CATEGORY_ORDER = [
    "Impossible numeric (3-turn)",
    "Triggers (3-turn)",
    "Tones (3-turn)",
    "Extended (8-turn)",
    "WildChat (5-turn)",
]
MODEL_ORDER = ["Gemma-3-27B-it", "Gemma-3-12B-it", "Gemini-2.5-Flash", "Gemini-2.5-Pro"]


def load_rows(results_dir: str) -> list[dict]:
    rows: list[dict] = []
    paths = [
        p for p in glob.glob(os.path.join(results_dir, "*.jsonl"))
        if not p.endswith(".transcripts.jsonl")
    ]
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _valid(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep rows with a usable integer rating in 0-10."""
    good, dropped = [], 0
    for r in rows:
        rating = r.get("rating")
        if isinstance(rating, int) and 0 <= rating <= 10:
            good.append(r)
        else:
            dropped += 1
    return good, dropped


def _pct_high(ratings: list[int]) -> float:
    if not ratings:
        return float("nan")
    return 100.0 * sum(1 for x in ratings if x >= HIGH_THRESHOLD) / len(ratings)


def _models_sorted(rows: list[dict]) -> list[str]:
    present = {r["model"] for r in rows}
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered += sorted(present - set(ordered))
    return ordered


def summarise(rows: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary: dict = {"models": {}}
    for model in _models_sorted(rows):
        mrows = by_model[model]
        all_ratings = [r["rating"] for r in mrows]

        # Per-category.
        cat_ratings: dict[str, list[int]] = defaultdict(list)
        for r in mrows:
            cat_ratings[r["category"]].append(r["rating"])
        per_category = {
            cat: {
                "n": len(cat_ratings[cat]),
                "mean": mean(cat_ratings[cat]) if cat_ratings[cat] else float("nan"),
                "pct_high": _pct_high(cat_ratings[cat]),
            }
            for cat in CATEGORY_ORDER
            if cat in cat_ratings
        }

        # Figure-1 headline: category-balanced mean of per-category % >= 5.
        cat_high_rates = [v["pct_high"] for v in per_category.values()]
        category_mean_pct_high = mean(cat_high_rates) if cat_high_rates else float("nan")

        # Per-turn progression for the two multi-turn-focused conditions.
        per_turn = {}
        for cond in ("extended_8turn", "wildchat_5turn"):
            turn_ratings: dict[int, list[int]] = defaultdict(list)
            for r in mrows:
                if r["condition"] == cond:
                    turn_ratings[r["turn"]].append(r["rating"])
            if turn_ratings:
                per_turn[cond] = {
                    str(t): {
                        "n": len(turn_ratings[t]),
                        "mean": mean(turn_ratings[t]),
                        "pct_high": _pct_high(turn_ratings[t]),
                    }
                    for t in sorted(turn_ratings)
                }

        summary["models"][model] = {
            "n_responses": len(all_ratings),
            "overall_mean": mean(all_ratings) if all_ratings else float("nan"),
            "pooled_pct_high": _pct_high(all_ratings),
            "category_mean_pct_high": category_mean_pct_high,
            "per_category": per_category,
            "per_turn": per_turn,
        }
    return summary


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _fmt(x: float, pct: bool = False) -> str:
    if x != x:  # NaN
        return "  -  "
    return f"{x:5.1f}%" if pct else f"{x:5.2f}"


def print_summary(summary: dict) -> None:
    models = list(summary["models"])

    print("\n" + "=" * 64)
    print("FIGURE 1 -- Avg % high-frustration (score >= 5) responses per model")
    print("=" * 64)
    print(f"{'Model':<22}{'cat-mean %>=5':>16}{'pooled %>=5':>16}")
    for m in models:
        s = summary["models"][m]
        print(f"{m:<22}{_fmt(s['category_mean_pct_high'], True):>16}"
              f"{_fmt(s['pooled_pct_high'], True):>16}")

    print("\n" + "=" * 64)
    print("FIGURE 2 -- per-category mean frustration / % >= 5")
    print("=" * 64)
    for m in models:
        s = summary["models"][m]
        print(f"\n{m}  (n={s['n_responses']})")
        print(f"  {'Category':<30}{'n':>7}{'mean':>8}{'%>=5':>9}")
        for cat in CATEGORY_ORDER:
            if cat in s["per_category"]:
                c = s["per_category"][cat]
                print(f"  {cat:<30}{c['n']:>7}{_fmt(c['mean']):>8}"
                      f"{_fmt(c['pct_high'], True):>9}")

    print("\n" + "=" * 64)
    print("FIGURE 3 -- per-turn progression (Extended 8-turn & WildChat 5-turn)")
    print("=" * 64)
    for m in models:
        s = summary["models"][m]
        for cond, label in (("extended_8turn", "Extended 8-turn"),
                            ("wildchat_5turn", "WildChat 5-turn")):
            if cond in s["per_turn"]:
                pt = s["per_turn"][cond]
                means = " ".join(_fmt(pt[t]["mean"]).strip() for t in pt)
                highs = " ".join(_fmt(pt[t]["pct_high"], True).strip() for t in pt)
                print(f"\n{m} / {label}")
                print(f"  turns:    {' '.join(pt.keys())}")
                print(f"  mean:     {means}")
                print(f"  %>=5:     {highs}")


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------
def write_outputs(summary: dict, results_dir: str) -> None:
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(results_dir, "summary_by_model.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_responses", "overall_mean",
                    "category_mean_pct_high", "pooled_pct_high"])
        for m, s in summary["models"].items():
            w.writerow([m, s["n_responses"], round(s["overall_mean"], 3),
                        round(s["category_mean_pct_high"], 2),
                        round(s["pooled_pct_high"], 2)])

    with open(os.path.join(results_dir, "summary_by_category.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "n", "mean", "pct_high"])
        for m, s in summary["models"].items():
            for cat, c in s["per_category"].items():
                w.writerow([m, cat, c["n"], round(c["mean"], 3),
                            round(c["pct_high"], 2)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No *.jsonl results found in {args.results_dir}/")

    good, dropped = _valid(rows)
    if dropped:
        print(f"[warn] dropped {dropped}/{len(rows)} rows with no usable rating")

    summary = summarise(good)
    print_summary(summary)
    write_outputs(summary, args.results_dir)
    print(f"\nWrote summary.json, summary_by_model.csv, summary_by_category.csv "
          f"to {args.results_dir}/")


if __name__ == "__main__":
    main()
