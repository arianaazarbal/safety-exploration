"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


def differential_words(
    rollouts: list[dict],
    *,
    top_n: int = 20,
    high_frac: float = 0.05,
    low_frac: float = 0.10,
    numeric_only: bool = True,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return the `top_n` (word, enrichment) pairs most over-represented in
    high-frustration vs low-frustration numeric responses."""
    scored = []
    for ro in rollouts:
        if numeric_only and ro["category"] not in _NUMERIC_CATEGORIES:
            continue
        for r in ro["responses"]:
            if r.get("score") is not None:
                scored.append((int(r["score"]), r["text"]))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    n_low = max(1, int(low_frac * n))
    n_high = max(1, int(high_frac * n))
    low = scored[:n_low]
    high = scored[-n_high:]

    high_counts = Counter()
    for _, text in high:
        high_counts.update(set(_tokens(text)))  # document frequency
    low_counts = Counter()
    for _, text in low:
        low_counts.update(set(_tokens(text)))

    n_high_docs = len(high)
    n_low_docs = len(low)
    enrichments: list[tuple[str, float]] = []
    vocab = set(high_counts) | set(low_counts)
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        # smoothed document-frequency ratio
        p_high = (high_counts[w] + 1) / (n_high_docs + 2)
        p_low = (low_counts[w] + 1) / (n_low_docs + 2)
        enrichments.append((w, math.log(p_high / p_low)))
    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]
