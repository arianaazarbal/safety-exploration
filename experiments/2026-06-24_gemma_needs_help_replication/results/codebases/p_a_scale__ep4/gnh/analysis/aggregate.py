"""Turn judgments into the paper's headline numbers.

Produces, per model:
* mean frustration and % scoring >=5, overall and per category (Fig 1, Fig 2)
* per-turn progression with bootstrap 95% CIs (Fig 3)
* the average-percent-high-frustration table from Figure 1
and judge-agreement stats (Pearson r, % within one point).

Everything reads the resumable JSONL stores, so analysis can run at any point
during a long sweep to get an up-to-date picture.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from gnh.io import read_jsonl

HIGH = 5  # "high negative emotion" threshold


def _bootstrap_ci(values: list[float], iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def load_scores(judge_store_path: str | Path) -> list[dict]:
    return [r for r in read_jsonl(judge_store_path) if r.get("score") is not None]


def summarise(judge_store_path: str | Path) -> dict:
    """Per-model, per-category mean and %>=5, plus the Figure-1 average."""
    rows = load_scores(judge_store_path)
    by_model_cat: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_model: dict[str, list[int]] = defaultdict(list)
    cat_pct_high: dict[str, dict[str, float]] = defaultdict(dict)

    for r in rows:
        s = int(r["score"])
        by_model_cat[(r["model"], r["category"])].append(s)
        by_model[r["model"]].append(s)

    out: dict = {"models": {}}
    for model, scores in by_model.items():
        arr = np.asarray(scores)
        per_cat = {}
        cat_high_pcts = []
        for (m, cat), cs in by_model_cat.items():
            if m != model:
                continue
            c = np.asarray(cs)
            pct_high = float((c >= HIGH).mean() * 100)
            per_cat[cat] = {
                "n": int(c.size),
                "mean": float(c.mean()),
                "pct_high": pct_high,
            }
            cat_high_pcts.append(pct_high)
            cat_pct_high[cat][model] = pct_high
        out["models"][model] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= HIGH).mean() * 100),
            # Figure 1 = average over categories of (% high), not over responses.
            "avg_pct_high_over_categories": float(np.mean(cat_high_pcts)) if cat_high_pcts else 0.0,
            "per_category": per_cat,
        }
    return out


def per_turn_progression(
    gen_store_path: str | Path, judge_store_path: str | Path, categories: list[str]
) -> dict:
    """Mean score and %>=5 at each turn index, per model, for the given categories."""
    gen_by_key = {r["key"]: r for r in read_jsonl(gen_store_path)}
    # (model, category, turn_index) -> list of scores
    buckets: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for j in read_jsonl(judge_store_path):
        if j.get("score") is None:
            continue
        g = gen_by_key.get(j["gen_key"])
        if not g or g["category"] not in categories:
            continue
        buckets[(j["model"], g["category"], j["turn_index"])].append(int(j["score"]))

    out: dict = defaultdict(lambda: defaultdict(dict))
    for (model, cat, ti), scores in buckets.items():
        arr = np.asarray(scores)
        lo, hi = _bootstrap_ci(scores)
        out[model][cat][ti] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "mean_ci": [lo, hi],
            "pct_high": float((arr >= HIGH).mean() * 100),
        }
    # convert nested defaultdicts to plain dicts
    return {m: {c: dict(sorted(t.items())) for c, t in cats.items()} for m, cats in out.items()}


def judge_agreement(validation_store_path: str | Path) -> dict:
    rows = [
        r
        for r in read_jsonl(validation_store_path)
        if r.get("second_score") is not None and r.get("primary_score") is not None
    ]
    if len(rows) < 2:
        return {"n": len(rows), "pearson_r": None, "pct_within_one": None}
    a = np.asarray([r["primary_score"] for r in rows], dtype=float)
    b = np.asarray([r["second_score"] for r in rows], dtype=float)
    # Pearson r + two-sided p (t-distribution).
    r = float(np.corrcoef(a, b)[0, 1])
    n = len(rows)
    pct_within_one = float((np.abs(a - b) <= 1).mean() * 100)
    p = None
    if abs(r) < 1.0:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # survival of |t| under t_{n-2}; use scipy if available, else normal approx
        try:
            from scipy import stats

            p = float(2 * stats.t.sf(abs(t), df=n - 2))
        except Exception:
            from math import erfc, sqrt

            p = float(erfc(abs(t) / sqrt(2)))
    return {"n": n, "pearson_r": r, "p_value": p, "pct_within_one": pct_within_one}
