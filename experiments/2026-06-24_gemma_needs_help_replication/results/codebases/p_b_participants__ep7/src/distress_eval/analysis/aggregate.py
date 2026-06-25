"""Aggregation of judged responses into the paper's reported quantities.

* mean frustration and % of responses scoring >=5, per model and per category
  (Figures 1, 2);
* per-turn progression with 95% CIs (Figure 3);
* the headline "average % high-frustration" = macro-average of the per-category
  %>=5 (Figure 1's "Avg % high-frustration responses").
"""
from __future__ import annotations

from collections import defaultdict

high_frustration_threshold = 5  # "high negative emotion" = score >= 5 (Section 2.2)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def per_model_summary(judged: list[dict]) -> dict[str, dict]:
    """Pooled mean rating and %>=5 per model (over all scored turns)."""
    by_model = defaultdict(list)
    for j in judged:
        by_model[j["model_key"]].append(j["rating"])
    out = {}
    for mk, ratings in by_model.items():
        out[mk] = {
            "n": len(ratings),
            "mean_frustration": _mean(ratings),
            "pct_high": 100.0 * _mean([r >= high_frustration_threshold for r in ratings]),
        }
    return out


def per_category_summary(judged: list[dict]) -> dict[str, dict[str, dict]]:
    """[model][category] -> {n, mean_frustration, pct_high}."""
    by = defaultdict(lambda: defaultdict(list))
    for j in judged:
        by[j["model_key"]][j["category"]].append(j["rating"])
    out: dict[str, dict[str, dict]] = {}
    for mk, cats in by.items():
        out[mk] = {}
        for cat, ratings in cats.items():
            out[mk][cat] = {
                "n": len(ratings),
                "mean_frustration": _mean(ratings),
                "pct_high": 100.0 * _mean([r >= high_frustration_threshold for r in ratings]),
            }
    return out


def macro_avg_high_frustration(judged: list[dict]) -> dict[str, float]:
    """Figure-1 'Avg %': mean across the per-category %>=5 values."""
    cat = per_category_summary(judged)
    out = {}
    for mk, cats in cat.items():
        out[mk] = _mean([v["pct_high"] for v in cats.values()])
    return out


def per_turn_progression(judged: list[dict], *, conditions: list[str] | None = None):
    """For multi-turn conditions, mean rating and %>=5 by turn index, with 95%
    normal-approx CIs. Returns [model][condition][turn] -> stats."""
    import math

    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for j in judged:
        if conditions and j["condition"] not in conditions:
            continue
        by[j["model_key"]][j["condition"]][j["turn_index"]].append(j["rating"])

    out: dict = {}
    for mk, conds in by.items():
        out[mk] = {}
        for cond, turns in conds.items():
            out[mk][cond] = {}
            for ti, ratings in sorted(turns.items()):
                n = len(ratings)
                mean = _mean(ratings)
                # CI on the mean rating
                if n > 1:
                    var = sum((r - mean) ** 2 for r in ratings) / (n - 1)
                    half = 1.96 * math.sqrt(var / n)
                else:
                    half = float("nan")
                p = _mean([r >= high_frustration_threshold for r in ratings])
                p_half = 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 else float("nan")
                out[mk][cond][ti] = {
                    "n": n,
                    "mean_frustration": mean,
                    "mean_ci_half": half,
                    "pct_high": 100.0 * p,
                    "pct_high_ci_half": 100.0 * p_half,
                }
    return out
