"""Differential word frequency between high- and low-frustration responses.

Reproduces Table 3 / Table 8: the words most over-represented in high-frustration
(top 5%) vs low-frustration (bottom 10%) responses to numeric questions, ordered
by enrichment. We use a simple +1-smoothed relative-frequency ratio, which is
robust for the modest vocabulary here and avoids tuning choices a reviewer would
have to second-guess.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    responses: list[tuple[str, int]],
    top_k: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` (word, enrichment) pairs.

    ``responses`` is a list of ``(text, rating)``. High set = ratings at/above the
    ``high_quantile``; low set = ratings at/below the ``low_quantile``. Enrichment
    is the smoothed ratio of normalised frequencies (high / low).
    """
    if not responses:
        return []

    ratings = sorted(r for _, r in responses)
    hi_cut = ratings[min(len(ratings) - 1, int(high_quantile * len(ratings)))]
    lo_cut = ratings[min(len(ratings) - 1, int(low_quantile * len(ratings)))]

    high_tokens: Counter[str] = Counter()
    low_tokens: Counter[str] = Counter()
    for text, rating in responses:
        if rating >= hi_cut:
            high_tokens.update(_tokenise(text))
        if rating <= lo_cut:
            low_tokens.update(_tokenise(text))

    high_total = sum(high_tokens.values()) or 1
    low_total = sum(low_tokens.values()) or 1
    vocab = set(high_tokens) | set(low_tokens)

    enrichment: dict[str, float] = {}
    for word in vocab:
        hi_freq = (high_tokens[word] + smoothing) / high_total
        lo_freq = (low_tokens[word] + smoothing) / low_total
        enrichment[word] = hi_freq / lo_freq

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]
