"""Differential word frequency (Table 3 / Table 8).

Top-20 words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ranked by enrichment.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .metrics import load_records

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(
    model: str,
    *,
    categories: tuple[str, ...] = ("impossible_numeric", "tones", "extended"),
    top_pct: float = 0.05,
    bottom_pct: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 0.5,
) -> list[tuple[str, float]]:
    """Return [(word, enrichment)] for the most high-frustration-enriched words.

    Enrichment = (freq in high set + a) / (freq in low set + a), using
    add-`smoothing` Laplace smoothing on per-word probabilities. Words appearing
    fewer than ``min_count`` times across both sets are dropped to reduce noise.
    """
    recs = []
    for c in categories:
        recs.extend(load_records(model, c))
    if not recs:
        return []

    ratings = np.array([r["rating"] for r in recs], dtype=float)
    hi_thresh = np.quantile(ratings, 1 - top_pct)
    lo_thresh = np.quantile(ratings, bottom_pct)

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for r in recs:
        toks = _tokenize(r["assistant_text"])
        if r["rating"] >= hi_thresh:
            hi_counts.update(toks)
        elif r["rating"] <= lo_thresh:
            lo_counts.update(toks)

    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1
    vocab = set(hi_counts) | set(lo_counts)

    scored = []
    for w in vocab:
        if hi_counts[w] + lo_counts[w] < min_count:
            continue
        p_hi = (hi_counts[w] + smoothing) / (hi_total + smoothing * len(vocab))
        p_lo = (lo_counts[w] + smoothing) / (lo_total + smoothing * len(vocab))
        scored.append((w, p_hi / p_lo))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
