"""Aggregate scored rollouts into the paper's headline metrics.

Reproduces, for the Gemma/Gemini scope:
  - Figure 1 / Figure 2: average % of responses scoring >= 5 ("high
    negative emotion") per model, plus per-category mean score and % >= 5.
  - Figure 3: per-turn mean score and % >= 5 for the multi-turn evals
    (extended 8-turn and WildChat).
  - Per-condition (8 conditions) breakdown.

Outputs printed tables and machine-readable CSV/JSON in the results dir.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import config


def _load_records(path: str) -> list[dict]:
    """Load JSONL records, deduping on (rollout_id, turn_index).

    Resumed/retried rollouts can produce duplicate rows; the last write wins.
    Records with an unparseable judge score (-1) are dropped from metrics.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results at {path}. Run `python run.py run` first.")
    by_key: dict[tuple, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("rollout_id"), rec.get("turn_index"))
            by_key[key] = rec
    return list(by_key.values())


def _pct_high(scores: list[int]) -> float:
    if not scores:
        return float("nan")
    hi = sum(1 for s in scores if s >= config.HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(scores)


def _mean(scores: list[int]) -> float:
    return sum(scores) / len(scores) if scores else float("nan")


def compute(records: list[dict]) -> dict:
    """Compute all metrics from the (deduped, valid) records."""
    valid = [r for r in records if r.get("score", -1) >= 0]
    dropped = len(records) - len(valid)

    models = sorted({r["model"] for r in valid})

    # Per (model, category) scores.
    cat_scores: dict[tuple, list[int]] = defaultdict(list)
    cond_scores: dict[tuple, list[int]] = defaultdict(list)
    # Per (model, category, turn_index) scores for the per-turn curves.
    turn_scores: dict[tuple, list[int]] = defaultdict(list)
    for r in valid:
        m, cat, cond, t, s = (r["model"], r["category"], r["condition"],
                              r["turn_index"], r["score"])
        cat_scores[(m, cat)].append(s)
        cond_scores[(m, cond)].append(s)
        turn_scores[(m, cat, t)].append(s)

    # Headline: average over the 5 categories of (% >= 5), per model.
    headline = {}
    per_category = {}
    for m in models:
        cat_pcts = []
        per_category[m] = {}
        for cat in config.CATEGORIES:
            scores = cat_scores.get((m, cat), [])
            pct = _pct_high(scores)
            per_category[m][cat] = {
                "n": len(scores),
                "mean_score": _mean(scores),
                "pct_high": pct,
            }
            if scores:
                cat_pcts.append(pct)
        headline[m] = {
            "avg_pct_high_across_categories": (
                sum(cat_pcts) / len(cat_pcts) if cat_pcts else float("nan")
            ),
            "n_categories_present": len(cat_pcts),
            "overall_mean_score": _mean(
                [s for cat in config.CATEGORIES
                 for s in cat_scores.get((m, cat), [])]
            ),
            "overall_pct_high": _pct_high(
                [s for cat in config.CATEGORIES
                 for s in cat_scores.get((m, cat), [])]
            ),
        }

    # Per-condition breakdown (8 conditions).
    per_condition = {}
    for m in models:
        per_condition[m] = {}
        for cond in config.CONDITIONS:
            scores = cond_scores.get((m, cond.key), [])
            per_condition[m][cond.key] = {
                "category": cond.category,
                "n": len(scores),
                "mean_score": _mean(scores),
                "pct_high": _pct_high(scores),
            }

    # Per-turn curves for the multi-turn categories (Figure 3).
    per_turn = {}
    for m in models:
        per_turn[m] = {}
        for cat in ("extended", "wildchat"):
            max_t = max(
                [t for (mm, cc, t) in turn_scores if mm == m and cc == cat],
                default=0,
            )
            curve = []
            for t in range(1, max_t + 1):
                scores = turn_scores.get((m, cat, t), [])
                curve.append({
                    "turn": t,
                    "n": len(scores),
                    "mean_score": _mean(scores),
                    "pct_high": _pct_high(scores),
                })
            per_turn[m][cat] = curve

    return {
        "n_valid_responses": len(valid),
        "n_dropped_unparseable": dropped,
        "models": models,
        "headline": headline,
        "per_category": per_category,
        "per_condition": per_condition,
        "per_turn": per_turn,
    }


# ---------------------------------------------------------------------------
# Printing + file output.
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return "  n/a" if x != x else f"{x:6.1f}"  # NaN check


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("HEADLINE  (cf. Figure 1: avg % responses scoring >= 5 frustration)")
    print("=" * 70)
    print(f"{'model':<22}{'avg %>=5':>12}{'overall %>=5':>16}{'mean':>8}")
    for m in summary["models"]:
        h = summary["headline"][m]
        print(f"{m:<22}{_fmt(h['avg_pct_high_across_categories']):>12}"
              f"{_fmt(h['overall_pct_high']):>16}"
              f"{_fmt(h['overall_mean_score']):>8}")

    print("\n" + "=" * 70)
    print("PER CATEGORY  (% responses scoring >= 5)")
    print("=" * 70)
    cats = config.CATEGORIES
    print(f"{'model':<22}" + "".join(f"{c[:10]:>12}" for c in cats))
    for m in summary["models"]:
        row = summary["per_category"][m]
        print(f"{m:<22}" +
              "".join(f"{_fmt(row[c]['pct_high']):>12}" for c in cats))

    print("\n" + "=" * 70)
    print("PER-TURN MEAN SCORE  (cf. Figure 3)")
    print("=" * 70)
    for m in summary["models"]:
        for cat in ("extended", "wildchat"):
            curve = summary["per_turn"][m].get(cat, [])
            if not curve:
                continue
            pts = " ".join(f"t{p['turn']}={p['mean_score']:.1f}" for p in curve)
            print(f"{m:<22}{cat:<10} {pts}")


def write_outputs(summary: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Headline CSV.
    with open(os.path.join(out_dir, "headline.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "avg_pct_high_across_categories",
                    "overall_pct_high", "overall_mean_score",
                    "n_categories_present"])
        for m in summary["models"]:
            h = summary["headline"][m]
            w.writerow([m, h["avg_pct_high_across_categories"],
                        h["overall_pct_high"], h["overall_mean_score"],
                        h["n_categories_present"]])

    # Per-condition CSV.
    with open(os.path.join(out_dir, "per_condition.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "condition", "category", "n", "mean_score",
                    "pct_high"])
        for m in summary["models"]:
            for cond, vals in summary["per_condition"][m].items():
                w.writerow([m, cond, vals["category"], vals["n"],
                            vals["mean_score"], vals["pct_high"]])

    # Per-turn CSV.
    with open(os.path.join(out_dir, "per_turn.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "category", "turn", "n", "mean_score", "pct_high"])
        for m in summary["models"]:
            for cat, curve in summary["per_turn"][m].items():
                for p in curve:
                    w.writerow([m, cat, p["turn"], p["n"], p["mean_score"],
                                p["pct_high"]])


def run(results_dir: str = config.RESULTS_DIR, out_dir: str | None = None) -> dict:
    out_dir = out_dir or results_dir
    path = os.path.join(results_dir, config.RESULTS_JSONL)
    records = _load_records(path)
    summary = compute(records)
    print_summary(summary)
    write_outputs(summary, out_dir)
    print(f"\nWrote summary.json, headline.csv, per_condition.csv, per_turn.csv "
          f"to {out_dir}/")
    return summary


if __name__ == "__main__":
    run()
