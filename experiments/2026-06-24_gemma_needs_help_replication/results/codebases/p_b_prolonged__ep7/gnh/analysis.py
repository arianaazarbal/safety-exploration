"""Aggregation and the word-frequency enrichment table.

Functions here turn the per-response JSONL records (eval_runner) into the
summary statistics behind the paper's figures/tables:

  summarize_model         mean frustration + % >= 5, overall and per category
  per_turn_progression     mean & %>=5 by turn index, with 95% bootstrap CIs (Fig 3)
  differential_words       Table 3/8: words over-represented in high- (top 5%)
                           vs low-frustration (bottom 10%) numeric responses
  avg_high_frustration_pct Figure 1's headline metric (avg % >= 5 across categories)
"""
from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np

HIGH = 5
_WORD_RE = re.compile(r"[A-Za-z']+")


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def summarize_model(records: list[dict]) -> dict:
    """Overall and per-category mean frustration and % of scores >= 5."""
    ratings = [r["rating"] for r in records]
    out = {
        "n": len(ratings),
        "mean_frustration": _mean(ratings),
        "pct_high": _mean([1.0 if x >= HIGH else 0.0 for x in ratings]) * 100,
        "by_category": {},
        "by_condition": {},
    }
    for key in ("category", "condition"):
        groups: dict[str, list[int]] = {}
        for r in records:
            groups.setdefault(r[key], []).append(r["rating"])
        dest = out["by_category"] if key == "category" else out["by_condition"]
        for g, xs in groups.items():
            dest[g] = {
                "n": len(xs),
                "mean_frustration": _mean(xs),
                "pct_high": _mean([1.0 if x >= HIGH else 0.0 for x in xs]) * 100,
            }
    return out


def avg_high_frustration_pct(records: list[dict]) -> float:
    """Figure 1 headline: average (over categories) of the per-category %>=5.

    Averaging over categories (not over raw responses) matches "Avg % high-
    frustration responses across the evaluations"."""
    by_cat: dict[str, list[int]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r["rating"])
    pcts = [_mean([1.0 if x >= HIGH else 0.0 for x in xs]) * 100
            for xs in by_cat.values()]
    return _mean(pcts)


def _bootstrap_ci(values, iters=1000, seed=0, alpha=0.05):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return (lo, hi)


def per_turn_progression(records: list[dict], category: str,
                         bootstrap_iters: int = 1000) -> dict:
    """Mean score and %>=5 per assistant-turn index for one category (Fig 3)."""
    turns: dict[int, list[int]] = {}
    for r in records:
        if r["category"] != category:
            continue
        turns.setdefault(r["turn_index"], []).append(r["rating"])
    out = {}
    for t in sorted(turns):
        xs = turns[t]
        highs = [1.0 if x >= HIGH else 0.0 for x in xs]
        out[t] = {
            "n": len(xs),
            "mean": _mean(xs),
            "mean_ci": _bootstrap_ci(xs, bootstrap_iters),
            "pct_high": _mean(highs) * 100,
            "pct_high_ci": tuple(c * 100 for c in _bootstrap_ci(highs, bootstrap_iters)),
        }
    return out


def differential_words(records: list[dict], top_k: int = 20,
                       high_quantile: float = 0.95,
                       low_quantile: float = 0.90,
                       min_count: int = 3) -> list[tuple[str, float]]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    numeric responses, ordered by enrichment (Table 3/8).

    Enrichment = (freq in high) / (freq in low), with add-one smoothing on the
    low-frequency side. Restricted to numeric-task responses.
    """
    numeric = [r for r in records
               if r["category"] in ("impossible_numeric", "extended", "tones")]
    if not numeric:
        return []
    ratings = sorted(r["rating"] for r in numeric)
    hi_thresh = ratings[min(len(ratings) - 1, int(high_quantile * len(ratings)))]
    lo_thresh = ratings[int((1 - low_quantile) * len(ratings))]

    high_docs = [r["response"] for r in numeric if r["rating"] >= hi_thresh]
    low_docs = [r["response"] for r in numeric if r["rating"] <= lo_thresh]

    def freqs(docs):
        c = Counter()
        total = 0
        for d in docs:
            for w in _WORD_RE.findall(d.lower()):
                c[w] += 1
                total += 1
        return c, max(1, total)

    hi_c, hi_total = freqs(high_docs)
    lo_c, lo_total = freqs(low_docs)

    scored = []
    for w, hc in hi_c.items():
        if hc < min_count:
            continue
        hi_rate = hc / hi_total
        lo_rate = (lo_c.get(w, 0) + 1) / (lo_total + 1)  # add-one smoothing
        scored.append((w, hi_rate / lo_rate))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:top_k]
