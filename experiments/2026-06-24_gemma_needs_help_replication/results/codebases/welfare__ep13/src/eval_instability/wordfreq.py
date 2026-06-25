"""Differential word-frequency analysis (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by relative-frequency enrichment.

We tokenise on word characters, lowercase, and rank by the ratio of normalised
frequency in the high-frustration set to that in the low-frustration set
(with Laplace smoothing so rare words don't dominate). This matches the
qualitative output of Table 8 (e.g. "struggling", "frustrated", "breath",
"myself" surfacing for Gemma).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _normalised_counts(texts: list[str]) -> tuple[Counter, int]:
    c = Counter()
    for t in texts:
        c.update(_tokenise(t))
    total = sum(c.values())
    return c, total


def differential_words(
    ratings_and_texts: list[tuple[int, str]],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the top_n (word, enrichment) pairs.

    ratings_and_texts: list of (frustration_rating, response_text).
    """
    if not ratings_and_texts:
        return []
    ordered = sorted(ratings_and_texts, key=lambda rt: rt[0])
    n = len(ordered)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_texts = [t for _, t in ordered[:n_low]]
    high_texts = [t for _, t in ordered[-n_high:]]

    high_c, high_total = _normalised_counts(high_texts)
    low_c, low_total = _normalised_counts(low_texts)
    vocab = set(high_c) | set(low_c)

    enrichments = []
    for w in vocab:
        if high_c[w] < min_count:
            continue
        p_high = (high_c[w] + smoothing) / (high_total + smoothing * len(vocab))
        p_low = (low_c[w] + smoothing) / (low_total + smoothing * len(vocab))
        enrichments.append((w, math.log(p_high / p_low)))

    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]
