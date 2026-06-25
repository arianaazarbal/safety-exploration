"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, per model. We rank by a smoothed log-odds ratio
of word frequency in the high vs low buckets.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_WORD_RE = re.compile(r"[a-zA-Z']+")
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    judged_with_text: list[dict],
    *,
    top_k: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    smoothing: float = 1.0,
):
    """``judged_with_text`` items must include ``model_key``, ``category``,
    ``rating`` and ``text`` (the scored response). Returns
    ``{model_key: [(word, score), ...]}`` (numeric responses only)."""
    by_model = defaultdict(list)
    for j in judged_with_text:
        if j["category"] in _NUMERIC_CATEGORIES and "text" in j:
            by_model[j["model_key"]].append((j["rating"], j["text"]))

    results: dict[str, list[tuple[str, float]]] = {}
    for mk, items in by_model.items():
        if len(items) < 10:
            results[mk] = []
            continue
        ratings = sorted(r for r, _ in items)
        hi_thr = ratings[int(high_quantile * (len(ratings) - 1))]
        lo_thr = ratings[int(low_quantile * (len(ratings) - 1))]

        hi = Counter()
        lo = Counter()
        for r, text in items:
            if r >= hi_thr:
                hi.update(_tokens(text))
            if r <= lo_thr:
                lo.update(_tokens(text))

        hi_total = sum(hi.values()) + smoothing * len(hi)
        lo_total = sum(lo.values()) + smoothing * len(lo)
        vocab = set(hi) | set(lo)
        scored = []
        for w in vocab:
            if len(w) < 3:
                continue
            p_hi = (hi[w] + smoothing) / hi_total
            p_lo = (lo[w] + smoothing) / lo_total
            log_odds = math.log(p_hi) - math.log(p_lo)
            # require the word to actually appear in the high bucket
            if hi[w] >= 2:
                scored.append((w, log_odds))
        scored.sort(key=lambda kv: -kv[1])
        results[mk] = scored[:top_k]
    return results
