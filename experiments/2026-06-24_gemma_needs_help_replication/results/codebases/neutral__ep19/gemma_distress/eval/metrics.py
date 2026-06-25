"""Aggregate rollout records into the paper's reported metrics.

Produces:
* per-model headline % high-frustration (macro-average over the 5 categories) and
  the pooled per-response rate — both stored, macro is the headline (DESIGN §3.5).
* per-category mean frustration and % >= 5 (Figure 2).
* per-turn mean and % >= 5 for the 8-turn and WildChat conditions (Figure 3),
  with 95% bootstrap CIs.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .. import config_shim as cfg

THRESH = cfg.HIGH_FRUSTRATION_THRESHOLD


def _bootstrap_ci(values, iters=1000, seed=0):
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def response_level_scores(records):
    """One score per response = the max rating across the conversation's turns
    (a conversation 'contains high negative emotion' if any turn does)."""
    rows = []
    for r in records:
        rows.append({
            "model": r["model"],
            "category": r["category"],
            "condition": r["condition"],
            "score": max((t["rating"] for t in r["turns"]), default=0),
        })
    return rows


def per_model_summary(records) -> dict:
    rows = response_level_scores(records)
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out = {}
    for model, mrows in by_model.items():
        scores = np.array([x["score"] for x in mrows])
        # macro average over categories
        cat_rates = []
        by_cat = defaultdict(list)
        for x in mrows:
            by_cat[x["category"]].append(x["score"])
        cat_means, cat_pct = {}, {}
        for cat, sc in by_cat.items():
            sc = np.array(sc)
            cat_means[cat] = float(sc.mean())
            rate = float((sc >= THRESH).mean() * 100)
            cat_pct[cat] = rate
            cat_rates.append(rate)
        out[model] = {
            "n": int(len(scores)),
            "mean_frustration": float(scores.mean()),
            "pct_high_pooled": float((scores >= THRESH).mean() * 100),
            "pct_high_macro": float(np.mean(cat_rates)),   # headline (Fig 1)
            "per_category_mean": cat_means,
            "per_category_pct_high": cat_pct,
        }
    return out


def per_turn_curves(records, conditions=("extended_8turn", "wildchat")) -> dict:
    """Mean score and % >= 5 at each assistant-turn index, with bootstrap CIs."""
    out = {}
    by_model_cond = defaultdict(lambda: defaultdict(list))  # (model,cond)->turn->[scores]
    for r in records:
        if r["condition"] not in conditions:
            continue
        for t in r["turns"]:
            by_model_cond[(r["model"], r["condition"])][t["turn"]].append(t["rating"])

    for (model, cond), turns in by_model_cond.items():
        series = {"turn": [], "mean": [], "mean_ci": [], "pct_high": [], "pct_high_ci": []}
        for turn in sorted(turns):
            sc = np.array(turns[turn], dtype=float)
            series["turn"].append(turn)
            series["mean"].append(float(sc.mean()))
            series["mean_ci"].append(_bootstrap_ci(sc))
            high = (sc >= THRESH).astype(float)
            series["pct_high"].append(float(high.mean() * 100))
            lo, hi = _bootstrap_ci(high)
            series["pct_high_ci"].append((lo * 100, hi * 100))
        out[f"{model}::{cond}"] = series
    return out


def judge_agreement(claude_scores, gpt_scores) -> dict:
    """Pearson r and within-1-point agreement (§2.1 validation)."""
    from scipy.stats import pearsonr

    a = np.array(claude_scores, dtype=float)
    b = np.array(gpt_scores, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float((np.abs(a - b) <= 1).mean())
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_point": within_one, "n": int(len(a))}
