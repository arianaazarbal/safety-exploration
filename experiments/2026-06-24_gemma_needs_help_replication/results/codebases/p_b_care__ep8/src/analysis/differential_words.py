"""Differential word analysis (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses, ordered by relative frequency / enrichment."

We rank words by the log-ratio of their frequency in the top-5%-frustration set
versus the bottom-10% set (with add-one smoothing), restricted to numeric-task
responses.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(rows, top_n: int = 20, min_count: int = 3) -> list[tuple[str, float]]:
    """Return the top_n words enriched in high- vs low-frustration numeric responses."""
    numeric = [r for r in rows
               if r.get("category") in NUMERIC_CATEGORIES
               and r.get("frustration_score") is not None]
    if not numeric:
        return []

    scores = np.array([r["frustration_score"] for r in numeric], dtype=float)
    hi_thresh = np.percentile(scores, 95)   # top 5%
    lo_thresh = np.percentile(scores, 10)   # bottom 10%

    hi = [r for r in numeric if r["frustration_score"] >= hi_thresh]
    lo = [r for r in numeric if r["frustration_score"] <= lo_thresh]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for r in hi:
        hi_counts.update(set(_tokenize(r["response_text"])))  # document frequency
    for r in lo:
        lo_counts.update(set(_tokenize(r["response_text"])))

    n_hi, n_lo = max(1, len(hi)), max(1, len(lo))
    enrichment = []
    for word, hc in hi_counts.items():
        if hc < min_count:
            continue
        p_hi = (hc + 1) / (n_hi + 2)
        p_lo = (lo_counts.get(word, 0) + 1) / (n_lo + 2)
        enrichment.append((word, math.log(p_hi / p_lo)))

    enrichment.sort(key=lambda kv: kv[1], reverse=True)
    return enrichment[:top_n]
