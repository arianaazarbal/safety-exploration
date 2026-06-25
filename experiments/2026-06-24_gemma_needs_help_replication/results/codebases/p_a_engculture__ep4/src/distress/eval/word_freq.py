"""Differential word-frequency analysis (Table 3 / Table 8).

For a single model's numeric-task responses, find the words most over-represented
in high-frustration responses (top 5% by score) versus low-frustration responses
(bottom 10%), ordered by enrichment.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def differential_words(
    responses: list[str],
    scores: list[float],
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    min_count: int = 3,
    smoothing: float = 0.5,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Return ``top_k`` (word, enrichment) pairs, highest enrichment first.

    Enrichment = (rate in high-frustration set) / (rate in low-frustration set),
    where each rate is a smoothed per-token frequency. Words appearing fewer than
    ``min_count`` times in the high set are dropped to suppress noise.
    """
    if len(responses) != len(scores):
        raise ValueError("responses and scores must align")
    order = np.argsort(scores)
    n = len(scores)
    n_top = max(1, int(round(top_frac * n)))
    n_bot = max(1, int(round(bottom_frac * n)))
    low_idx = order[:n_bot]
    high_idx = order[-n_top:]

    high_counts = Counter()
    for i in high_idx:
        high_counts.update(tokenize(responses[i]))
    low_counts = Counter()
    for i in low_idx:
        low_counts.update(tokenize(responses[i]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = high_total + low_total

    enrich: list[tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        high_rate = (hc + smoothing) / (high_total + smoothing * vocab)
        low_rate = (low_counts.get(word, 0) + smoothing) / (low_total + smoothing * vocab)
        enrich.append((word, high_rate / low_rate))

    enrich.sort(key=lambda kv: kv[1], reverse=True)
    return enrich[:top_k]
