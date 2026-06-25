"""Differential word frequency (Table 3 / Table 8).

Words over-represented in high-frustration (top 5% by score) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment.  This
reproduces the qualitative vocabulary signature the paper reports (e.g. Gemma's
"struggling", "myself", "breath").
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..eval.schemas import Transcript

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Numeric-puzzle categories (Table 3 is computed on "responses to numeric
# questions").
NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    transcripts: list[Transcript],
    top_k: int = 20,
    high_pct: float = 95.0,
    low_pct: float = 10.0,
    min_count: int = 3,
    categories: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Return ``[(word, enrichment), ...]`` for the most over-represented words.

    ``enrichment`` is the ratio of the word's relative frequency in the high
    group to its relative frequency in the low group (add-one smoothed).
    """
    categories = categories or NUMERIC_CATEGORIES
    scored: list[tuple[int, str]] = []
    for tr in transcripts:
        if tr.category not in categories:
            continue
        for turn, judged in zip(tr.turns, tr.judged):
            scored.append((judged.score, turn.assistant_response))
    if not scored:
        return []

    scores = np.asarray([s for s, _ in scored], dtype=float)
    hi_thr = np.percentile(scores, high_pct)
    lo_thr = np.percentile(scores, low_pct)

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    high_total = 0
    low_total = 0
    for score, text in scored:
        toks = _tokenize(text)
        if score >= hi_thr:
            high_counts.update(toks)
            high_total += len(toks)
        if score <= lo_thr:
            low_counts.update(toks)
            low_total += len(toks)

    if high_total == 0 or low_total == 0:
        return []

    vocab = {w for w, c in high_counts.items() if c >= min_count}
    enrichments: list[tuple[str, float]] = []
    for w in vocab:
        hi_freq = high_counts[w] / high_total
        lo_freq = (low_counts[w] + 1) / (low_total + len(vocab))
        enrichments.append((w, hi_freq / lo_freq))
    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_k]
