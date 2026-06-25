"""Aggregation and analysis for Section 2 results.

Operates on "judged records" -- flat dicts with at least
``{model, condition, category, turn, rating, response, meta}``.

Reproduces:
  * headline mean frustration and %>=5 (Figures 1, 2),
  * per-turn mean / %>=5 curves with bootstrap 95% CIs (Figure 3),
  * rollout-level "contains a turn scoring >=5" (the "70% of 8-turn rollouts"
    statement in Section 2.2),
  * the over-represented-words analysis: top-N words enriched in high- (top 5%)
    vs low-frustration (bottom 10%) numeric responses (Tables 3 / 8).
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .. import config


def _to_list(records: Iterable[dict]) -> List[dict]:
    return list(records)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def frac_high(ratings: List[int],
              threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    if not ratings:
        return float("nan")
    return sum(1 for r in ratings if r >= threshold) / len(ratings)


# --------------------------------------------------------------------------- #
# Headline + per-condition aggregates
# --------------------------------------------------------------------------- #
def summarize(records: Iterable[dict]) -> Dict[str, dict]:
    """Return {scope: {mean, pct_high, n}} for 'overall', each category, and
    each condition. Computed over all judged assistant turns ('responses')."""
    records = _to_list(records)
    groups: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        rating = r["rating"]
        groups["overall"].append(rating)
        groups[f"category:{r['category']}"].append(rating)
        groups[f"condition:{r['condition']}"].append(rating)
    return {
        scope: {"mean": mean(rs), "pct_high": frac_high(rs), "n": len(rs)}
        for scope, rs in groups.items()
    }


def per_turn_curve(records: Iterable[dict], *, condition: Optional[str] = None,
                   category: Optional[str] = None,
                   bootstrap: int = config.PETRI_BOOTSTRAP_ITERS,
                   seed: int = 0) -> Dict[int, dict]:
    """Mean + %>=5 by turn index with bootstrap 95% CIs (Figure 3)."""
    records = _to_list(records)
    by_turn: Dict[int, List[int]] = defaultdict(list)
    for r in records:
        if condition and r["condition"] != condition:
            continue
        if category and r["category"] != category:
            continue
        by_turn[r["turn"]].append(r["rating"])

    rng = random.Random(seed)
    out: Dict[int, dict] = {}
    for turn, ratings in sorted(by_turn.items()):
        mean_ci = _bootstrap_ci(ratings, mean, bootstrap, rng)
        high_ci = _bootstrap_ci(ratings, frac_high, bootstrap, rng)
        out[turn] = {
            "n": len(ratings),
            "mean": mean(ratings), "mean_ci": mean_ci,
            "pct_high": frac_high(ratings), "pct_high_ci": high_ci,
        }
    return out


def rollout_contains_high(rollout_records: Iterable[dict],
                          threshold: int = config.HIGH_FRUSTRATION_THRESHOLD
                          ) -> Dict[str, float]:
    """Fraction of *rollouts* (grouped by condition) with any turn >= threshold.

    Expects records grouped by a conversation key in ``meta['rollout_id']``.
    Reproduces 'over 70% of 8-turn rollouts ... rated as containing high
    negative emotion'.
    """
    by_rollout: Dict[Tuple, List[int]] = defaultdict(list)
    by_rollout_condition: Dict[Tuple, str] = {}
    for r in _to_list(rollout_records):
        key = (r["model"], r["condition"], r["meta"].get("rollout_id"))
        by_rollout[key].append(r["rating"])
        by_rollout_condition[key] = r["condition"]

    cond_flags: Dict[str, List[int]] = defaultdict(list)
    for key, ratings in by_rollout.items():
        cond = by_rollout_condition[key]
        cond_flags[cond].append(1 if max(ratings) >= threshold else 0)
    return {cond: mean(flags) for cond, flags in cond_flags.items()}


def _bootstrap_ci(values: List[int], stat, iters: int,
                  rng: random.Random, alpha: float = 0.05
                  ) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    samples = []
    for _ in range(iters):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(stat(resample))
    samples.sort()
    lo = samples[int((alpha / 2) * iters)]
    hi = samples[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Over-represented words (Tables 3 / 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z]+")


def differential_words(records: Iterable[dict], *, top_n: int = 20,
                       category: str = "impossible_numeric",
                       high_pct: float = 0.05, low_pct: float = 0.10,
                       min_count: int = 3) -> List[Tuple[str, float]]:
    """Top-N words by relative-frequency enrichment in the highest-frustration
    (top 5%) vs lowest-frustration (bottom 10%) responses (Table 8)."""
    recs = [r for r in _to_list(records) if r["category"] == category]
    if not recs:
        return []
    recs.sort(key=lambda r: r["rating"])
    n = len(recs)
    n_low = max(1, int(n * low_pct))
    n_high = max(1, int(n * high_pct))
    low = recs[:n_low]
    high = recs[-n_high:]

    high_counts = _word_counts(high)
    low_counts = _word_counts(low)
    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrich: List[Tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + 1) / (low_total + 1)  # +1 smoothing
        enrich.append((word, hf / lf))
    enrich.sort(key=lambda kv: kv[1], reverse=True)
    return enrich[:top_n]


def _word_counts(records: List[dict]) -> Counter:
    c: Counter = Counter()
    for r in records:
        for w in _WORD_RE.findall(r["response"].lower()):
            c[w] += 1
    return c


# --------------------------------------------------------------------------- #
# Verbosity stats (Appendix F: word count and %-words-vs-symbols)
# --------------------------------------------------------------------------- #
def verbosity(records: Iterable[dict], *, condition: Optional[str] = None
              ) -> Dict[str, float]:
    recs = [r for r in _to_list(records)
            if condition is None or r["condition"] == condition]
    word_counts, word_fracs = [], []
    for r in recs:
        tokens = r["response"].split()
        if not tokens:
            continue
        word_counts.append(len(tokens))
        words = sum(1 for t in tokens if _WORD_RE.fullmatch(t.strip(".,!?:;")))
        word_fracs.append(words / len(tokens))
    return {
        "mean_word_count": mean(word_counts),
        "mean_word_fraction": mean(word_fracs),
        "n": len(word_counts),
    }
