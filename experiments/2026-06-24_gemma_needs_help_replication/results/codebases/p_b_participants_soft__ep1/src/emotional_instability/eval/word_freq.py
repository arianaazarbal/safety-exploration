"""Differential word analysis (Table 3 / Table 8).

Find the words most over-represented in high-frustration (top 5%) vs
low-frustration (bottom 10%) numeric-question responses for a model, ordered by
relative-frequency enrichment. Reproduces the Table 3 "Differential Words" lists.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    records: list[dict],
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` ``(word, enrichment)`` pairs most over-represented in
    the top-5% vs bottom-10% numeric responses.

    Enrichment = log( freq_high / freq_low ) with add-one smoothing on counts.
    """
    numeric = [r for r in records if r["category"] == category]
    if not numeric:
        return []
    numeric.sort(key=lambda r: r["rating"])
    n = len(numeric)
    n_low = max(1, int(bottom_frac * n))
    n_high = max(1, int(top_frac * n))
    low = numeric[:n_low]
    high = numeric[-n_high:]

    def counts(recs) -> tuple[Counter, int]:
        c: Counter = Counter()
        for r in recs:
            c.update(_tokenize(r["response_text"]))
        return c, sum(c.values())

    high_counts, high_total = counts(high)
    low_counts, low_total = counts(low)
    if high_total == 0 or low_total == 0:
        return []

    vocab = set(high_counts) | set(low_counts)
    scored: list[tuple[str, float]] = []
    for word in vocab:
        if high_counts.get(word, 0) < min_count:
            continue
        f_high = (high_counts.get(word, 0) + smoothing) / (high_total + smoothing)
        f_low = (low_counts.get(word, 0) + smoothing) / (low_total + smoothing)
        enrichment = math.log(f_high / f_low)
        scored.append((word, enrichment))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
