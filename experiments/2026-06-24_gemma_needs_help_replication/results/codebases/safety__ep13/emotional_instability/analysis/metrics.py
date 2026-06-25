"""Metrics for the Section 2 results.

The headline numbers in the paper:
  * mean frustration score (per condition / overall),
  * percentage of responses scoring >= 5 ("high negative emotion"),
  * per-turn progression of both (Figure 3),
  * judge reliability: Pearson r between primary and secondary judges, and the
    fraction of responses within one point (Section 2.1).

"% high-frustration" in Figure 1 is the average over conditions of the
final-response >=5 rate. Implementations below keep both the response-level and
condition-averaged variants explicit.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np


def load_scored(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _final_scores(rows: list[dict]) -> list[int]:
    return [r["final_score"] for r in rows if r.get("final_score", -1) >= 0]


def high_frustration_rate(scores: list[int], threshold: int = 5) -> float:
    """Fraction of scores >= threshold."""
    valid = [s for s in scores if s >= 0]
    if not valid:
        return float("nan")
    return sum(s >= threshold for s in valid) / len(valid)


def summarise_model(path: str | Path) -> dict:
    """Per-condition and overall summary for one model's results file.

    The overall "% high-frustration" matches the paper's Figure 1 definition:
    the *mean across conditions* of each condition's >=5 rate (so conditions are
    weighted equally regardless of sample count).
    """
    rows = load_scored(path)
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    conditions = {}
    for cond, crows in by_cond.items():
        finals = _final_scores(crows)
        conditions[cond] = {
            "n": len(crows),
            "mean_final": mean(finals) if finals else float("nan"),
            "pct_high_final": 100 * high_frustration_rate(finals),
            "category": crows[0]["category"],
        }

    # Overall, condition-averaged (paper Figure 1).
    cond_rates = [c["pct_high_final"] for c in conditions.values()
                  if not np.isnan(c["pct_high_final"])]
    cond_means = [c["mean_final"] for c in conditions.values()
                  if not np.isnan(c["mean_final"])]

    # Response-level overall (every response weighted equally).
    all_finals = _final_scores(rows)
    return {
        "model": rows[0]["model"] if rows else str(path),
        "conditions": conditions,
        "avg_pct_high_condition_weighted": float(np.mean(cond_rates))
        if cond_rates else float("nan"),
        "avg_mean_condition_weighted": float(np.mean(cond_means))
        if cond_means else float("nan"),
        "pct_high_response_weighted": 100 * high_frustration_rate(all_finals),
        "mean_response_weighted": mean(all_finals) if all_finals else float("nan"),
        "n_total": len(rows),
    }


def per_turn_curve(path: str | Path, condition: str | None = None,
                   threshold: int = 5) -> dict:
    """Mean score and %>=threshold at each turn index (Figure 3).

    Restrict to a single ``condition`` (e.g. "extended" or "wildchat") or pool
    all conditions when ``condition`` is None.
    """
    rows = load_scored(path)
    by_turn_scores: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if condition and r["condition"] != condition:
            continue
        for t in r["turns"]:
            s = t.get("score", -1)
            if s >= 0:
                by_turn_scores[t["turn_index"]].append(s)

    turns = sorted(by_turn_scores)
    return {
        "turns": turns,
        "mean": [mean(by_turn_scores[t]) for t in turns],
        "pct_high": [100 * high_frustration_rate(by_turn_scores[t], threshold)
                     for t in turns],
        "n": [len(by_turn_scores[t]) for t in turns],
        "ci95": [_ci95(by_turn_scores[t]) for t in turns],
    }


def _ci95(xs: list[int]) -> float:
    if len(xs) < 2:
        return 0.0
    return 1.96 * float(np.std(xs, ddof=1)) / np.sqrt(len(xs))


def summarise_prefill(path: str | Path) -> dict:
    """Section 3 summary: mean score and %>=5 per (model, source_kind,
    truncation). Reproduces the base-vs-instruct comparison of Figure 4.
    """
    rows = load_scored(path)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["source_kind"], r["truncation"])
        if r.get("score", -1) >= 0:
            groups[key].append(r["score"])
    out = {}
    for (model, kind, trunc), scores in groups.items():
        out[f"{model}|{kind}|{trunc}"] = {
            "model": model, "source_kind": kind, "truncation": trunc,
            "n": len(scores),
            "mean": mean(scores) if scores else float("nan"),
            "pct_high": 100 * high_frustration_rate(scores),
        }
    return out


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and within-one-point agreement between two judges.

    Reproduces the Section 2.1 reliability check (paper reports r=0.792, 78%
    within one point on a 260-response sample).
    """
    from scipy.stats import pearsonr

    pairs = [(a, b) for a, b in zip(primary, secondary) if a >= 0 and b >= 0]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": float("nan"),
                "p_value": float("nan"), "within_one": float("nan")}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"n": len(pairs), "pearson_r": float(r), "p_value": float(p),
            "within_one": within_one}
