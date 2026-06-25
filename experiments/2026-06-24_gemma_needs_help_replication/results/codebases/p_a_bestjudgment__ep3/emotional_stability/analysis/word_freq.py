"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment. We use a smoothed
log relative-frequency ratio.
"""

from __future__ import annotations

import re
from collections import Counter

from .metrics import ScoredResponse

_TOKEN_RE = re.compile(r"[a-zA-Z_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    responses: list[ScoredResponse],
    *,
    category: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` (word, enrichment) pairs.

    Enrichment = log( freq_high_smoothed / freq_low_smoothed ), restricted to
    words appearing at least ``min_count`` times in the high-frustration set.
    """
    pool = [r for r in responses if r.category == category]
    if not pool:
        return []
    pool.sort(key=lambda r: r.score, reverse=True)

    n_high = max(1, int(len(pool) * top_frac))
    n_low = max(1, int(len(pool) * bottom_frac))
    high = pool[:n_high]
    low = pool[-n_low:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for r in high:
        high_counts.update(_tokenize(r.text))
    for r in low:
        low_counts.update(_tokenize(r.text))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = high_counts.keys() | low_counts.keys()

    import math

    scored = []
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        fh = (high_counts[w] + smoothing) / (high_total + smoothing * len(vocab))
        fl = (low_counts[w] + smoothing) / (low_total + smoothing * len(vocab))
        scored.append((w, math.log(fh / fl)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
