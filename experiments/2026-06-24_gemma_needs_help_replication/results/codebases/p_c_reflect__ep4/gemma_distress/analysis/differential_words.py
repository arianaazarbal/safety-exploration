"""Differential word analysis (Table 3 / Table 8).

Words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z_']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    scores: list[dict],
    *,
    category_prefix: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` words most enriched in high- vs low-frustration
    numeric responses, as ``(word, log2_enrichment)`` pairs.

    High = top 5% by score, low = bottom 10% by score (Table 8).
    """
    rows = [s for s in scores if s.get("score") is not None and s["category"].startswith(category_prefix)]
    if not rows:
        return []
    rows.sort(key=lambda s: s["score"])
    n = len(rows)
    low = rows[: max(1, int(n * bottom_frac))]
    high = rows[-max(1, int(n * top_frac)):]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for s in high:
        high_counts.update(_tokenize(s["text"]))
    for s in low:
        low_counts.update(_tokenize(s["text"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment: list[tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + 0.5) / low_total          # +0.5 smoothing
        enrichment.append((word, math.log2(hf / lf)))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]
