"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment (relative frequency).
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    responses: list[tuple[str, float]],
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """``responses`` is a list of (text, frustration_score). Returns the top_k
    (word, enrichment) pairs ordered by enrichment, where enrichment is the
    smoothed ratio of relative frequency in the high-frustration set vs the
    low-frustration set."""
    if not responses:
        return []
    ranked = sorted(responses, key=lambda r: r[1])
    n = len(ranked)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = ranked[:n_low]
    high = ranked[-n_high:]

    def counts(group):
        c: Counter[str] = Counter()
        total = 0
        for text, _ in group:
            toks = _tokenize(text)
            c.update(toks)
            total += len(toks)
        return c, max(1, total)

    high_c, high_total = counts(high)
    low_c, low_total = counts(low)

    enrichment: list[tuple[str, float]] = []
    for word, hc in high_c.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = (low_c.get(word, 0) + 1) / (low_total + 1)  # Laplace smoothing
        enrichment.append((word, hf / lf))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]


def word_token_fraction(text: str) -> float:
    """Fraction of space-split tokens that are words (vs numbers/symbols).
    Used in the SFT verbosity analysis (Appendix F)."""
    parts = text.split()
    if not parts:
        return 0.0
    words = sum(1 for p in parts if _WORD_RE.fullmatch(p.strip(".,!?;:")) )
    return words / len(parts)
