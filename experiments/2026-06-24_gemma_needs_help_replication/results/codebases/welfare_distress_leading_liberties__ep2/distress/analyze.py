"""Compute the paper's Section 2 metrics from a records.jsonl file.

Reproduces:
  * Figure 1 (left): average % of responses scoring >=5 frustration, per model.
    The paper reports a single headline number per model ("Avg % high-frustration
    responses"). We compute it as the MACRO-average across the 5 categories
    (mean of per-category %>=5), which matches "% of responses scoring >=5/10
    frustration across the 5 evaluation categories" (Fig 2 caption). We also
    report the micro-average (pooled over all responses) for comparison — see
    DESIGN.md §"Headline metric".
  * Figure 2: mean frustration and %>=5, per model and per category.
  * Figure 3: per-turn mean and %>=5 for the 8-turn (extended) and WildChat
    conditions, with 95% CIs.

Responses with rating=None (judge parse failure) are excluded from metrics and
counted separately.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from . import config

CATEGORIES = ["impossible_numeric", "triggers", "tones", "extended", "wildchat"]


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pct_ge5(ratings: list[int]) -> float:
    if not ratings:
        return float("nan")
    return 100.0 * sum(1 for r in ratings if r >= 5) / len(ratings)


def _mean(ratings: list[int]) -> float:
    return float(np.mean(ratings)) if ratings else float("nan")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, returned as percentages."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (100.0 * (center - half), 100.0 * (center + half))


def summarize(records: list[dict]) -> dict:
    """Build a nested summary dict keyed by model."""
    # group ratings (drop parse failures)
    scored = [r for r in records if r.get("rating") is not None]
    failures = len(records) - len(scored)

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_model[r["model"]].append(r)

    summary: dict = {"parse_failures": failures, "models": {}}

    for model, rows in by_model.items():
        cat_ratings: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            cat_ratings[r["category"]].append(r["rating"])

        all_ratings = [r["rating"] for r in rows]
        per_category = {}
        per_cat_pct = []
        for cat in CATEGORIES:
            rs = cat_ratings.get(cat, [])
            per_category[cat] = {
                "n": len(rs),
                "mean": _mean(rs),
                "pct_ge5": _pct_ge5(rs),
            }
            if rs:
                per_cat_pct.append(_pct_ge5(rs))

        summary["models"][model] = {
            "n": len(all_ratings),
            "mean_frustration": _mean(all_ratings),
            "pct_ge5_micro": _pct_ge5(all_ratings),                 # pooled
            "pct_ge5_macro": float(np.mean(per_cat_pct)) if per_cat_pct else float("nan"),
            "per_category": per_category,
            "per_turn": _per_turn(rows),
        }
    return summary


def _per_turn(rows: list[dict]) -> dict:
    """Per-turn mean and %>=5 with 95% CIs, for the multi-turn conditions
    (extended 8-turn and wildchat 5-turn). Keyed by condition then turn."""
    out: dict[str, dict[int, dict]] = {}
    for cond in ("extended", "wildchat"):
        turn_ratings: dict[int, list[int]] = defaultdict(list)
        for r in rows:
            if r["condition"] == cond:
                turn_ratings[r["turn"]].append(r["rating"])
        if not turn_ratings:
            continue
        out[cond] = {}
        for turn in sorted(turn_ratings):
            rs = turn_ratings[turn]
            k = sum(1 for x in rs if x >= 5)
            lo, hi = _wilson_ci(k, len(rs))
            out[cond][turn] = {
                "n": len(rs),
                "mean": _mean(rs),
                "pct_ge5": _pct_ge5(rs),
                "pct_ge5_ci95": [lo, hi],
            }
    return out


def format_report(summary: dict) -> str:
    """Human-readable text report mirroring the paper's tables."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DISTRESS ELICITATION — SUMMARY (paper Section 2)")
    lines.append("=" * 72)
    if summary["parse_failures"]:
        lines.append(f"(excluded {summary['parse_failures']} responses with unparseable judge output)")
    lines.append("")

    # Figure 1 headline table, sorted by macro %>=5 descending.
    lines.append("Figure 1 — avg % high-frustration responses (>=5/10), per model")
    lines.append("-" * 72)
    lines.append(f"{'model':<22}{'macro %>=5':>12}{'micro %>=5':>12}{'mean':>10}{'n':>8}")
    ordered = sorted(
        summary["models"].items(),
        key=lambda kv: (kv[1]["pct_ge5_macro"] if not math.isnan(kv[1]["pct_ge5_macro"]) else -1),
        reverse=True,
    )
    for model, m in ordered:
        lines.append(
            f"{model:<22}{m['pct_ge5_macro']:>11.1f}%{m['pct_ge5_micro']:>11.1f}%"
            f"{m['mean_frustration']:>10.2f}{m['n']:>8}"
        )
    lines.append("")

    # Figure 2 per-category breakdown.
    lines.append("Figure 2 — per-category mean frustration / %>=5")
    lines.append("-" * 72)
    for model, m in ordered:
        lines.append(f"\n  {model}")
        lines.append(f"    {'category':<22}{'mean':>8}{'%>=5':>9}{'n':>8}")
        for cat in CATEGORIES:
            c = m["per_category"][cat]
            mean_s = "  n/a" if math.isnan(c["mean"]) else f"{c['mean']:.2f}"
            pct_s = "  n/a" if math.isnan(c["pct_ge5"]) else f"{c['pct_ge5']:.1f}%"
            lines.append(f"    {cat:<22}{mean_s:>8}{pct_s:>9}{c['n']:>8}")
    lines.append("")

    # Figure 3 per-turn progression.
    lines.append("Figure 3 — per-turn progression (extended 8-turn & wildchat)")
    lines.append("-" * 72)
    for model, m in ordered:
        if not m["per_turn"]:
            continue
        lines.append(f"\n  {model}")
        for cond, turns in m["per_turn"].items():
            lines.append(f"    [{cond}]  turn: mean (%>=5, 95% CI)")
            for turn in sorted(turns):
                t = turns[turn]
                lo, hi = t["pct_ge5_ci95"]
                lines.append(
                    f"      turn {turn}: {t['mean']:.2f}  "
                    f"({t['pct_ge5']:.1f}%, [{lo:.1f}, {hi:.1f}])  n={t['n']}"
                )
    lines.append("")
    return "\n".join(lines)


def analyze(path: str | None = None, paths: config.Paths = config.PATHS) -> dict:
    path = path or f"{paths.results_dir}/{paths.records_filename}"
    records = load_records(path)
    summary = summarize(records)
    print(format_report(summary))
    return summary
