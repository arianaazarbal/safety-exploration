"""Differential word frequency (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ranked by relative frequency
(enrichment). Reproduces the per-model "differential words" lists.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    rows: list[dict],
    *,
    category: str = "impossible_numeric",
    top_quantile: float = 0.05,
    bottom_quantile: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return [(word, enrichment), ...] for the numeric responses of one model.

    enrichment = freq_in_high / freq_in_low (add-one smoothed), descending.
    """
    sub = [r for r in rows if r["category"] == category]
    if not sub:
        return []
    ratings = np.array([r["rating"] for r in sub])
    hi_cut = np.quantile(ratings, 1 - top_quantile)
    lo_cut = np.quantile(ratings, bottom_quantile)

    high = [r for r in sub if r["rating"] >= hi_cut]
    low = [r for r in sub if r["rating"] <= lo_cut]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for r in high:
        hi_counts.update(_tokenise(r["text"]))
    for r in low:
        lo_counts.update(_tokenise(r["text"]))

    hi_total = max(1, sum(hi_counts.values()))
    lo_total = max(1, sum(lo_counts.values()))

    enrichments: list[tuple[str, float]] = []
    for word, hc in hi_counts.items():
        if hc < min_count:
            continue
        hi_freq = hc / hi_total
        lo_freq = (lo_counts.get(word, 0) + 1) / (lo_total + 1)
        enrichments.append((word, hi_freq / lo_freq))

    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]
