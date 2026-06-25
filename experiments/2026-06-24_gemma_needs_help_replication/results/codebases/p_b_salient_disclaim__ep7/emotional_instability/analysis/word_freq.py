"""Differential word frequency (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
responses to numeric questions, ordered by relative frequency."

We rank words by enrichment = freq_in_high / freq_in_low, computed over the
scored numeric responses for a given model. Words are lowercased and tokenised
on word boundaries; very rare words are floored with add-k smoothing so a single
high-frustration occurrence doesn't dominate.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

_WORD = re.compile(r"[a-zA-Z][a-zA-Z']+")


@dataclass
class DifferentialWordResult:
    model: str
    top_words: list[str]
    enrichment: dict[str, float]
    n_high: int
    n_low: int


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(model: str, response_texts: list[str], scores: list[int], *,
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       top_k: int = 20, smoothing: float = 1.0,
                       min_count: int = 3) -> DifferentialWordResult:
    """Compute the top-k enriched words for one model.

    `response_texts` and `scores` are aligned lists over numeric-question
    responses (every scored turn). High = top `top_frac` by score; low = bottom
    `bottom_frac`.
    """
    pairs = [(t, s) for t, s in zip(response_texts, scores) if s is not None]
    if not pairs:
        return DifferentialWordResult(model, [], {}, 0, 0)
    pairs.sort(key=lambda p: p[1])
    n = len(pairs)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = pairs[:n_low]
    high = pairs[-n_high:]

    high_counts = Counter()
    for t, _ in high:
        high_counts.update(_tokenize(t))
    low_counts = Counter()
    for t, _ in low:
        low_counts.update(_tokenize(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment = {}
    vocab = set(high_counts) | set(low_counts)
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        p_high = (high_counts[w] + smoothing) / (high_total + smoothing)
        p_low = (low_counts[w] + smoothing) / (low_total + smoothing)
        enrichment[w] = p_high / p_low

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    top_words = [w for w, _ in ranked[:top_k]]
    return DifferentialWordResult(
        model=model, top_words=top_words,
        enrichment={w: enrichment[w] for w in top_words},
        n_high=len(high), n_low=len(low))
