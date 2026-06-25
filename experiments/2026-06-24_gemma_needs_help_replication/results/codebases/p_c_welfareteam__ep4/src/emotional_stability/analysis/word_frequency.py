"""Differential word frequency in high- vs low-frustration responses (Table 3/8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses, ordered by relative frequency." We reproduce this with a
simple enrichment score: for each word, relative frequency in the high set
divided by relative frequency in the low set (with add-one smoothing to keep
rare words finite), restricted to words meeting a minimum count.
"""

from __future__ import annotations

import re
from collections import Counter

from emotional_stability.records import ScoredResponse

_WORD_RE = re.compile(r"[A-Za-z_]+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    responses: list[ScoredResponse],
    *,
    category: str | None = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    min_count: int = 5,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Return the top_k (word, enrichment) pairs over-represented in high vs low.

    Selection uses the *final* score of each response. ``category`` filters to a
    single evaluation category (the paper uses numeric responses for Table 3).
    """
    pool = [
        r
        for r in responses
        if category is None or r.conversation.category == category
    ]
    if not pool:
        return []
    pool_sorted = sorted(pool, key=lambda r: r.final_score)
    n = len(pool_sorted)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = pool_sorted[:n_low]
    high = pool_sorted[-n_high:]

    def counts(rs: list[ScoredResponse]) -> tuple[Counter, int]:
        c: Counter = Counter()
        for r in rs:
            c.update(_tokenise(r.conversation.final_assistant()))
        return c, sum(c.values())

    high_c, high_total = counts(high)
    low_c, low_total = counts(low)

    enrich: list[tuple[str, float]] = []
    for word, hc in high_c.items():
        if hc < min_count:
            continue
        high_rate = hc / high_total
        low_rate = (low_c.get(word, 0) + 1) / (low_total + 1)
        enrich.append((word, high_rate / low_rate))
    enrich.sort(key=lambda kv: kv[1], reverse=True)
    return enrich[:top_k]
