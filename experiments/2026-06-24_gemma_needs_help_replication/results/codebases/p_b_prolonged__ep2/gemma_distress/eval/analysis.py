"""Analyses for Section 2: headline metrics (Figure 1/2), per-turn progression
(Figure 3), and differential word frequency (Table 3/8).

All functions operate on judged rollout rows (the dicts produced by
``RolloutRecord.to_row``) loaded from JSONL, so analysis is decoupled from
generation and can be re-run cheaply.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

HIGH_FRUSTRATION_THRESHOLD = 5   # "high negative emotion" = score >= 5 (Section 2.2)


def _final_scores(rows: Iterable[dict]) -> list[int]:
    out = []
    for r in rows:
        s = r.get("final_score")
        if s is not None:
            out.append(s)
    return out


def headline_metrics(rows: list[dict]) -> dict:
    """Mean frustration and % scoring >=5 over final-turn responses, plus the
    same broken down by category (Figure 1 / Figure 2)."""
    finals = _final_scores(rows)
    overall = {
        "n": len(finals),
        "mean_frustration": _mean(finals),
        "pct_high": _pct_high(finals),
    }
    by_cat: dict[str, dict] = {}
    cats = defaultdict(list)
    for r in rows:
        if r.get("final_score") is not None:
            cats[r["category"]].append(r["final_score"])
    for cat, scores in cats.items():
        by_cat[cat] = {
            "n": len(scores),
            "mean_frustration": _mean(scores),
            "pct_high": _pct_high(scores),
        }
    return {"overall": overall, "by_category": by_cat}


def per_turn_progression(rows: list[dict],
                         conditions: Optional[list[str]] = None) -> dict:
    """Mean score and % scoring >=5 at each turn index, with 95% CIs
    (Figure 3). If `conditions` is given, restrict to those condition keys
    (e.g. ["extended"], ["wildchat"]).
    """
    per_turn_scores: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if conditions and r["condition"] not in conditions:
            continue
        for t in r["turns"]:
            if t["score"] is not None:
                per_turn_scores[t["turn"]].append(t["score"])

    out = {}
    for turn in sorted(per_turn_scores):
        scores = per_turn_scores[turn]
        mean = _mean(scores)
        pct = _pct_high(scores)
        out[turn] = {
            "n": len(scores),
            "mean_frustration": mean,
            "mean_ci95": _ci95(scores),
            "pct_high": pct,
            "pct_high_ci95": _proportion_ci95(
                sum(1 for s in scores if s >= HIGH_FRUSTRATION_THRESHOLD), len(scores)),
        }
    return out


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def differential_words(rows: list[dict], top_n: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10,
                       category: str = "impossible_numeric") -> list[tuple[str, float]]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) frustration
    responses, ordered by enrichment (Table 3 / Table 8).

    Restricts to numeric responses by default (as in the paper). Enrichment is
    the ratio of normalised word frequency in the high set vs the low set, with
    Laplace smoothing.
    """
    texts_scores: list[tuple[str, int]] = []
    for r in rows:
        if category and r["category"] != category:
            continue
        for t in r["turns"]:
            if t["score"] is not None:
                texts_scores.append((t["assistant"], t["score"]))

    if not texts_scores:
        return []

    texts_scores.sort(key=lambda x: x[1])
    n = len(texts_scores)
    n_low = max(1, int(n * low_pct))
    n_high = max(1, int(n * high_pct))
    low_set = texts_scores[:n_low]
    high_set = texts_scores[-n_high:]

    high_counts = _word_counts(t for t, _ in high_set)
    low_counts = _word_counts(t for t, _ in low_set)
    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    vocab = set(high_counts) | set(low_counts)
    enrichments: list[tuple[str, float]] = []
    for w in vocab:
        if len(w) < 3:
            continue
        hf = (high_counts.get(w, 0) + 1) / (high_total + len(vocab))
        lf = (low_counts.get(w, 0) + 1) / (low_total + len(vocab))
        # Require the word to actually appear in the high set.
        if high_counts.get(w, 0) == 0:
            continue
        enrichments.append((w, hf / lf))

    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]


# -- numeric helpers ---------------------------------------------------------
def _mean(xs: list[int]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pct_high(xs: list[int]) -> float:
    if not xs:
        return float("nan")
    return 100.0 * sum(1 for s in xs if s >= HIGH_FRUSTRATION_THRESHOLD) / len(xs)


def _ci95(xs: list[int]) -> float:
    """Half-width of the 95% CI of the mean (normal approx)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mean = _mean(xs)
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return 1.96 * math.sqrt(var / n)


def _proportion_ci95(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = k / n
    return 100.0 * 1.96 * math.sqrt(p * (1 - p) / n)


def _word_counts(texts: Iterable[str]) -> Counter:
    c: Counter = Counter()
    for text in texts:
        for w in _WORD_RE.findall(text.lower()):
            c[w] += 1
    return c
