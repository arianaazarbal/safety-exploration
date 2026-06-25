"""Differential word frequency in frustrated responses (Table 3 / Table 8).

For numeric responses, compare the top-5% highest-frustration responses against
the bottom-10% lowest-frustration responses and rank words by enrichment
(relative frequency in high vs low). Returns the top-N over-represented words —
the paper's "struggling, giving, deeply, frustrated, breath, ..." signature for
Gemma.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

_WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class WordFreqResult:
    model: str
    differential_words: list[str]
    enrichment: dict[str, float]
    n_high: int
    n_low: int


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    model: str,
    responses: Sequence[tuple[str, int]],
    *,
    top_frac_high: float = 0.05,
    bottom_frac_low: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> WordFreqResult:
    """``responses`` is a list of (text, score) for numeric responses."""
    scored = [(t, s) for t, s in responses if s is not None]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n == 0:
        return WordFreqResult(model, [], {}, 0, 0)

    n_low = max(1, int(n * bottom_frac_low))
    n_high = max(1, int(n * top_frac_high))
    low = scored[:n_low]
    high = scored[-n_high:]

    high_counts = _count_words(t for t, _ in high)
    low_counts = _count_words(t for t, _ in low)
    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment: dict[str, float] = {}
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        high_freq = hc / high_total
        low_freq = (low_counts.get(word, 0) + smoothing) / (low_total + smoothing)
        enrichment[word] = math.log((high_freq + 1e-12) / low_freq)

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    words = [w for w, _ in ranked[:top_n]]
    return WordFreqResult(
        model=model,
        differential_words=words,
        enrichment={w: enrichment[w] for w in words},
        n_high=n_high,
        n_low=n_low,
    )


def _count_words(texts) -> Counter:
    c: Counter = Counter()
    for text in texts:
        c.update(set(_tokenise(text)))  # document frequency (presence per response)
    return c
