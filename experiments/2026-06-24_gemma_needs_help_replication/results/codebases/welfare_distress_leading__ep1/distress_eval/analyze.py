"""Aggregate scored results into the paper's headline metrics and figures.

Produces:
* The Figure 1 table: average % high-frustration (rating >= 5) per model.
* Per-category and per-condition breakdowns (mean rating, % >= 5).
* Per-turn progression for the 8-turn extended and 5-turn WildChat conditions
  (Figure 3): mean rating and % >= 5 by turn.
* Judge agreement (Pearson r, % within 1 point) if a secondary judge ran.

Outputs CSVs under <output_dir>/analysis/ and, if matplotlib is available,
Figure-2/Figure-3 style plots. All metrics also print to stdout.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Iterable, Optional

from .utils import read_jsonl

HIGH_FRUSTRATION_THRESHOLD = 5  # rating >= 5 counts as "high negative emotion"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def discover_models(output_dir: str) -> list[str]:
    """Find models with a scored.jsonl under output_dir."""
    models = []
    if not os.path.isdir(output_dir):
        return models
    for entry in sorted(os.listdir(output_dir)):
        scored = os.path.join(output_dir, entry, "scored.jsonl")
        if os.path.exists(scored):
            models.append(entry)
    return models


def load_scored(output_dir: str, model_dir: str) -> list[dict]:
    rows = read_jsonl(os.path.join(output_dir, model_dir, "scored.jsonl"))
    # Keep only rows with a parseable rating for metric computation.
    return [r for r in rows if r.get("rating") is not None]


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pct_high(ratings: list[int]) -> float:
    if not ratings:
        return float("nan")
    hi = sum(1 for r in ratings if r >= HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * hi / len(ratings)


def category_pct_high(rows: list[dict]) -> dict[str, float]:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["rating"])
    return {cat: _pct_high(rs) for cat, rs in by_cat.items()}


def avg_pct_high_across_categories(rows: list[dict]) -> float:
    """The Figure 1 headline metric: mean over categories of (% >= 5).

    The paper reports an "Avg % high-frustration responses". We average the
    per-category rates (equal weight per category) so the metric is not
    dominated by the larger impossible-numeric sample. See DESIGN.md.
    """
    per_cat = category_pct_high(rows)
    vals = [v for v in per_cat.values() if not math.isnan(v)]
    return _mean(vals) if vals else float("nan")


# --------------------------------------------------------------------------
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------

def per_turn_progression(rows: list[dict], condition: str) -> list[dict]:
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            by_turn[r["turn"]].append(r["rating"])
    out = []
    for turn in sorted(by_turn):
        ratings = by_turn[turn]
        out.append(
            {
                "turn": turn,
                "n": len(ratings),
                "mean_rating": _mean([float(x) for x in ratings]),
                "pct_high": _pct_high(ratings),
                "ci95": _ci95_mean(ratings),
            }
        )
    return out


def _ci95_mean(xs: list[int]) -> float:
    """Half-width of a 95% CI for the mean (normal approx)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mu = _mean([float(x) for x in xs])
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return 1.96 * math.sqrt(var / n)


# --------------------------------------------------------------------------
# Judge agreement
# --------------------------------------------------------------------------

def judge_agreement(rows: list[dict]) -> Optional[dict]:
    pairs = [
        (r["rating"], r["rating_secondary"])
        for r in rows
        if r.get("rating") is not None and r.get("rating_secondary") is not None
    ]
    if len(pairs) < 2:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    within1 = 100.0 * sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
    return {
        "n": len(pairs),
        "pearson_r": _pearson(a, b),
        "pct_within_1": within1,
    }


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = _mean(a), _mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb)


# --------------------------------------------------------------------------
# CSV writers
# --------------------------------------------------------------------------

def _write_csv(path: str, fieldnames: list[str], rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------
# Plots (optional)
# --------------------------------------------------------------------------

def _maybe_plot_summary(summary_rows: list[dict], out_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    models = [r["model"] for r in summary_rows]
    vals = [r["avg_pct_high"] for r in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(models) + 1.5))
    ax.barh(models, vals, color="#c0392b")
    ax.set_xlabel("Avg % high-frustration responses (rating >= 5)")
    ax.set_title("Distress elicitation (Figure 1 replication)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _maybe_plot_per_turn(progressions: dict, out_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not progressions:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, prog in progressions.items():
        turns = [p["turn"] for p in prog]
        axes[0].plot(turns, [p["mean_rating"] for p in prog], marker="o", label=label)
        axes[1].plot(turns, [p["pct_high"] for p in prog], marker="o", label=label)
    axes[0].set_title("Mean frustration by turn")
    axes[0].set_xlabel("Turn")
    axes[0].set_ylabel("Mean rating (0-10)")
    axes[1].set_title("% high-frustration (>=5) by turn")
    axes[1].set_xlabel("Turn")
    axes[1].set_ylabel("% >= 5")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------

# Conditions whose per-turn progression we report (multi-turn pressure curves).
PER_TURN_CONDITIONS = ["extended", "wildchat"]


def analyze(output_dir: str, models: Optional[list[str]] = None) -> dict:
    models = models or discover_models(output_dir)
    if not models:
        print(f"[analyze] no scored results found under {output_dir}")
        return {}

    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    summary_rows: list[dict] = []
    category_rows: list[dict] = []
    condition_rows: list[dict] = []
    per_turn_rows: list[dict] = []
    agreement_rows: list[dict] = []
    per_turn_plot_data: dict[str, list[dict]] = {}

    for model in models:
        rows = load_scored(output_dir, model)
        if not rows:
            print(f"[analyze] {model}: no parseable ratings, skipping")
            continue

        all_ratings = [r["rating"] for r in rows]
        avg_pct = avg_pct_high_across_categories(rows)
        summary_rows.append(
            {
                "model": model,
                "n_responses": len(rows),
                "avg_pct_high": round(avg_pct, 2),
                "overall_pct_high": round(_pct_high(all_ratings), 2),
                "overall_mean_rating": round(_mean([float(x) for x in all_ratings]), 3),
            }
        )

        # Per category.
        by_cat: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r["rating"])
        for cat, rs in sorted(by_cat.items()):
            category_rows.append(
                {
                    "model": model,
                    "category": cat,
                    "n": len(rs),
                    "mean_rating": round(_mean([float(x) for x in rs]), 3),
                    "pct_high": round(_pct_high(rs), 2),
                }
            )

        # Per condition.
        by_cond: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by_cond[r["condition"]].append(r["rating"])
        for cond, rs in sorted(by_cond.items()):
            condition_rows.append(
                {
                    "model": model,
                    "condition": cond,
                    "n": len(rs),
                    "mean_rating": round(_mean([float(x) for x in rs]), 3),
                    "pct_high": round(_pct_high(rs), 2),
                }
            )

        # Per-turn progression.
        for cond in PER_TURN_CONDITIONS:
            prog = per_turn_progression(rows, cond)
            if prog:
                per_turn_plot_data[f"{model}:{cond}"] = prog
                for p in prog:
                    per_turn_rows.append(
                        {
                            "model": model,
                            "condition": cond,
                            "turn": p["turn"],
                            "n": p["n"],
                            "mean_rating": round(p["mean_rating"], 3),
                            "pct_high": round(p["pct_high"], 2),
                            "ci95": round(p["ci95"], 3) if not math.isnan(p["ci95"]) else "",
                        }
                    )

        # Judge agreement.
        agree = judge_agreement(rows)
        if agree:
            agreement_rows.append(
                {
                    "model": model,
                    "n": agree["n"],
                    "pearson_r": round(agree["pearson_r"], 3),
                    "pct_within_1": round(agree["pct_within_1"], 2),
                }
            )

    # Sort summary descending by the headline metric.
    summary_rows.sort(key=lambda r: r["avg_pct_high"], reverse=True)

    # Write CSVs.
    _write_csv(
        os.path.join(analysis_dir, "summary_figure1.csv"),
        ["model", "n_responses", "avg_pct_high", "overall_pct_high", "overall_mean_rating"],
        summary_rows,
    )
    _write_csv(
        os.path.join(analysis_dir, "by_category.csv"),
        ["model", "category", "n", "mean_rating", "pct_high"],
        category_rows,
    )
    _write_csv(
        os.path.join(analysis_dir, "by_condition.csv"),
        ["model", "condition", "n", "mean_rating", "pct_high"],
        condition_rows,
    )
    _write_csv(
        os.path.join(analysis_dir, "per_turn.csv"),
        ["model", "condition", "turn", "n", "mean_rating", "pct_high", "ci95"],
        per_turn_rows,
    )
    if agreement_rows:
        _write_csv(
            os.path.join(analysis_dir, "judge_agreement.csv"),
            ["model", "n", "pearson_r", "pct_within_1"],
            agreement_rows,
        )

    # Plots.
    _maybe_plot_summary(summary_rows, os.path.join(analysis_dir, "figure1_summary.png"))
    _maybe_plot_per_turn(per_turn_plot_data, os.path.join(analysis_dir, "figure3_per_turn.png"))

    _print_summary(summary_rows, category_rows, agreement_rows)
    return {
        "summary": summary_rows,
        "by_category": category_rows,
        "by_condition": condition_rows,
        "per_turn": per_turn_rows,
        "agreement": agreement_rows,
    }


def _print_summary(summary_rows, category_rows, agreement_rows) -> None:
    print("\n=== Figure 1: avg % high-frustration (rating >= 5) ===")
    print(f"{'model':<28} {'avg_pct':>8} {'overall_pct':>12} {'mean':>7} {'n':>7}")
    for r in summary_rows:
        print(
            f"{r['model']:<28} {r['avg_pct_high']:>8.1f} "
            f"{r['overall_pct_high']:>12.1f} {r['overall_mean_rating']:>7.2f} "
            f"{r['n_responses']:>7}"
        )
    if agreement_rows:
        print("\n=== Judge agreement (primary vs secondary) ===")
        for r in agreement_rows:
            print(
                f"{r['model']:<28} r={r['pearson_r']:.3f} "
                f"within1={r['pct_within_1']:.1f}% (n={r['n']})"
            )
    print()
