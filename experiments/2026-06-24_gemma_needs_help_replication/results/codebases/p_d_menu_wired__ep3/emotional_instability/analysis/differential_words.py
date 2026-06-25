"""Table 3 / Table 8: words over-represented in high- (top 5%) vs low-
frustration (bottom 10%) numeric responses, ordered by enrichment.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text or "") if len(w) > 1]


def differential_words(responses: list[dict], *, top_n: int = 20,
                       category: str = "impossible_numeric",
                       min_count: int = 3) -> list[tuple[str, float]]:
    """``responses`` is a flat list of ``{"score": int, "response": str,
    "category": str}``. Returns the top ``top_n`` (word, enrichment) pairs.

    Enrichment = log-ratio of a word's frequency in the top-5%-frustration set
    vs the bottom-10% set (with add-one smoothing).
    """
    pool = [r for r in responses if r.get("category", category) == category]
    if not pool:
        pool = responses
    scores = sorted(r["score"] for r in pool)
    if not scores:
        return []
    n = len(scores)
    hi_cut = scores[max(int(math.ceil(0.95 * n)) - 1, 0)]
    lo_cut = scores[min(int(math.floor(0.10 * n)), n - 1)]

    hi_docs = [r["response"] for r in pool if r["score"] >= hi_cut]
    lo_docs = [r["response"] for r in pool if r["score"] <= lo_cut]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for d in hi_docs:
        hi_counts.update(set(_tokenise(d)))  # document frequency
    for d in lo_docs:
        lo_counts.update(set(_tokenise(d)))

    hi_total = max(len(hi_docs), 1)
    lo_total = max(len(lo_docs), 1)

    enrichments = []
    for word, hc in hi_counts.items():
        if hc < min_count:
            continue
        hi_freq = hc / hi_total
        lo_freq = (lo_counts.get(word, 0) + 1) / (lo_total + 1)
        enrichments.append((word, math.log(hi_freq / lo_freq)))

    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]
