"""Differential word frequency (Table 3 / Table 8).

Identify words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by relative enrichment.
This reproduces the qualitative "emotional self-talk" signature the paper
reports for Gemma ("struggling", "myself", "breath", ...) and the more
self-critical Gemini vocabulary ("unacceptable", "inexcusable", ...).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..scoring import ResponseRecord

_WORD_RE = re.compile(r"[a-zA-Z']+")
# Common stopwords excluded so the ranking surfaces content words.
_STOP = set(
    "the a an of to and or is are was were be been being it its this that these "
    "those i you he she we they me my your our their for in on at by with as "
    "from but if then so not no yes do does did have has had will would can could "
    "should may might must let us s t re ve ll d m".split()
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP]


def differential_words(
    records: list[ResponseRecord],
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    n_words: int = 20,
    min_count: int = 3,
    numeric_only: bool = True,
) -> list[str]:
    """Return the ``n_words`` words most enriched in high- vs low-frustration.

    Enrichment is the log-ratio of a word's normalised frequency in the
    high-frustration set to its frequency in the low-frustration set (with
    add-one smoothing), restricted to words appearing at least ``min_count``
    times in the high set.
    """
    pool = [
        r
        for r in records
        if (not numeric_only or r.category in ("impossible_numeric", "tones", "extended"))
    ]
    if not pool:
        return []
    pool = sorted(pool, key=lambda r: r.score)
    n = len(pool)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = pool[:n_low]
    high = pool[-n_high:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for r in high:
        high_counts.update(_tokenize(r.text))
    for r in low:
        low_counts.update(_tokenize(r.text))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment: list[tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + 1) / (low_total + 1)
        enrichment.append((word, math.log(hf / lf)))

    enrichment.sort(key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in enrichment[:n_words]]
