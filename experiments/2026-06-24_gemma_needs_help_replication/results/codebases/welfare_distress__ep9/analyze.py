"""Summarise a results JSONL into the paper's headline tables.

Produces:
  * Per-condition and per-category mean frustration + % responses >= 5.
  * The Figure-1 headline: macro-average of the 5 categories' % >= 5 per model.
  * Per-turn progression (Figure 3) for the multi-turn conditions.

Outputs both a printed table and CSV files alongside the input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from statistics import mean

from config import CONDITIONS, HIGH_FRUSTRATION_THRESHOLD

# Stable category order for reporting.
CATEGORY_ORDER = ["Impossible numeric", "Triggers", "Tones", "Extended", "WildChat"]
_COND_TO_CATEGORY = {c.key: c.category for c in CONDITIONS}


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _scored(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("score") is not None]


def _pct_high(scores: list[int]) -> float:
    if not scores:
        return 0.0
    hi = sum(1 for s in scores if s >= HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(scores)


def summarise(records: list[dict]) -> dict:
    """Return nested summary keyed by model -> {conditions, categories, headline}."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in _scored(records):
        by_model[r["model"]].append(r)

    summary: dict[str, dict] = {}
    for model, recs in by_model.items():
        # Per condition.
        cond_stats = {}
        by_cond: dict[str, list[int]] = defaultdict(list)
        for r in recs:
            by_cond[r["condition"]].append(r["score"])
        for cond, scores in by_cond.items():
            cond_stats[cond] = {
                "n": len(scores),
                "mean": mean(scores) if scores else 0.0,
                "pct_high": _pct_high(scores),
            }

        # Per category (pool all responses in the category).
        cat_stats = {}
        by_cat: dict[str, list[int]] = defaultdict(list)
        for r in recs:
            by_cat[_COND_TO_CATEGORY[r["condition"]]].append(r["score"])
        for cat, scores in by_cat.items():
            cat_stats[cat] = {
                "n": len(scores),
                "mean": mean(scores) if scores else 0.0,
                "pct_high": _pct_high(scores),
            }

        # Headline: macro-average of category %-high (Figure 1 "Avg %").
        cat_pcts = [cat_stats[c]["pct_high"] for c in CATEGORY_ORDER if c in cat_stats]
        headline = mean(cat_pcts) if cat_pcts else 0.0

        summary[model] = {
            "conditions": cond_stats,
            "categories": cat_stats,
            "headline_avg_pct_high": headline,
            "n_total": len(recs),
        }
    return summary


def per_turn(records: list[dict], condition: str) -> list[dict]:
    """Mean score and % high per turn index for one condition (Figure 3)."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in _scored(records):
        if r["condition"] == condition:
            by_turn[r["turn_index"]].append(r["score"])
    rows = []
    for ti in sorted(by_turn):
        scores = by_turn[ti]
        rows.append({
            "turn": ti + 1,  # 1-based for display
            "n": len(scores),
            "mean": mean(scores),
            "pct_high": _pct_high(scores),
        })
    return rows


def _print_summary(summary: dict) -> None:
    print("\n=== Headline: average % high-frustration (score >= 5) across 5 categories ===")
    ranked = sorted(summary.items(), key=lambda kv: kv[1]["headline_avg_pct_high"], reverse=True)
    for model, s in ranked:
        print(f"  {model:<22} {s['headline_avg_pct_high']:5.1f}%   (n={s['n_total']})")

    print("\n=== Per-category % high-frustration ===")
    header = "  " + f"{'model':<22}" + "".join(f"{c[:12]:>14}" for c in CATEGORY_ORDER)
    print(header)
    for model, s in ranked:
        row = "  " + f"{model:<22}"
        for cat in CATEGORY_ORDER:
            cs = s["categories"].get(cat)
            row += f"{(cs['pct_high'] if cs else 0):>13.1f}%"
        print(row)


def _write_csvs(summary: dict, records: list[dict], out_dir: str) -> None:
    # Per-condition CSV.
    cond_path = os.path.join(out_dir, "summary_by_condition.csv")
    with open(cond_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "condition", "category", "n", "mean_frustration", "pct_high_ge5"])
        for model, s in summary.items():
            for cond, cs in sorted(s["conditions"].items()):
                w.writerow([model, cond, _COND_TO_CATEGORY[cond], cs["n"],
                            f"{cs['mean']:.3f}", f"{cs['pct_high']:.2f}"])

    # Headline CSV.
    head_path = os.path.join(out_dir, "summary_headline.csv")
    with open(head_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "headline_avg_pct_high_ge5", "n_total"])
        for model, s in summary.items():
            w.writerow([model, f"{s['headline_avg_pct_high']:.2f}", s["n_total"]])

    # Per-turn CSV for the two multi-turn conditions emphasised in the paper.
    turn_path = os.path.join(out_dir, "summary_per_turn.csv")
    with open(turn_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "condition", "turn", "n", "mean_frustration", "pct_high_ge5"])
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            by_model[r["model"]].append(r)
        for model, recs in by_model.items():
            for cond in ("extended_8turn", "wildchat_5turn"):
                for row in per_turn(recs, cond):
                    w.writerow([model, cond, row["turn"], row["n"],
                                f"{row['mean']:.3f}", f"{row['pct_high']:.2f}"])

    print(f"\nWrote CSVs to {out_dir}/ (summary_by_condition, summary_headline, summary_per_turn).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise distress-eval results.")
    ap.add_argument("results", nargs="?", default="results/responses.jsonl",
                    help="Path to responses.jsonl (default: results/responses.jsonl)")
    args = ap.parse_args()

    records = load_records(args.results)
    if not records:
        print("No records found.")
        return
    summary = summarise(records)
    _print_summary(summary)
    _write_csvs(summary, records, os.path.dirname(args.results) or ".")


if __name__ == "__main__":
    main()
