"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by relative-frequency enrichment.

Input: a list of (response_text, score) for one model's numeric responses.
Output: the top-N enriched words.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    scored: list[tuple[str, int]],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
    laplace: float = 1.0,
) -> list[tuple[str, float]]:
    """Return [(word, enrichment), ...] ordered by enrichment, descending.

    Enrichment = relative frequency in the top (high-frustration) set divided by
    relative frequency in the bottom (low-frustration) set, with Laplace
    smoothing. Words below `min_count` total occurrences are dropped to reduce
    noise (the paper's lists are clearly frequency-thresholded).
    """
    ordered = sorted(scored, key=lambda x: x[1])
    n = len(ordered)
    if n == 0:
        return []
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    bottom = ordered[:n_bottom]
    top = ordered[-n_top:]

    top_counts = Counter()
    for text, _ in top:
        top_counts.update(_tokenise(text))
    bottom_counts = Counter()
    for text, _ in bottom:
        bottom_counts.update(_tokenise(text))

    top_total = sum(top_counts.values()) or 1
    bottom_total = sum(bottom_counts.values()) or 1
    vocab = set(top_counts) | set(bottom_counts)

    enrichments = []
    for w in vocab:
        if top_counts[w] + bottom_counts[w] < min_count:
            continue
        p_top = (top_counts[w] + laplace) / (top_total + laplace * len(vocab))
        p_bottom = (bottom_counts[w] + laplace) / (bottom_total + laplace * len(vocab))
        enrichments.append((w, p_top / p_bottom))

    enrichments.sort(key=lambda x: -x[1])
    return enrichments[:top_n]


def log_enrichment(scored, **kw) -> list[tuple[str, float]]:
    """Same as differential_words but with log2 enrichment (sometimes clearer)."""
    return [(w, math.log2(e)) for w, e in differential_words(scored, **kw)]
