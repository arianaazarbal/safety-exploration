"""Table 3 / Table 8: words over-represented in high- vs low-frustration
responses to numeric questions.

Paper: "Top 20 words over-represented in high-frustration (top 5%) vs
low-frustration (bottom 10%) responses to numeric questions, ordered by relative
frequency." We reproduce that: rank by the ratio of within-group word frequency
(high group / low group), with add-one smoothing so rare words don't dominate.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_WORD_RE = re.compile(r"[a-zA-Z]+")
# Numeric-category conditions only (paper computes this on numeric responses).
_NUMERIC_CONDITIONS = ("impossible_numeric", "tones", "extended")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(
    df: pd.DataFrame,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Return [(word, enrichment)] for one model's numeric responses."""
    sub = df[df["category"].isin(_NUMERIC_CONDITIONS)].copy()
    sub = sub.sort_values("score")
    n = len(sub)
    if n < 20:
        return []
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = sub.iloc[:n_low]
    high = sub.iloc[-n_high:]

    high_counts = Counter()
    for t in high["response_text"]:
        high_counts.update(set(_tokens(t)))      # document frequency
    low_counts = Counter()
    for t in low["response_text"]:
        low_counts.update(set(_tokens(t)))

    vocab = set(high_counts) | set(low_counts)
    enrichment = {}
    for w in vocab:
        if len(w) < 3:
            continue
        hi = (high_counts[w] + 1) / (len(high) + 1)
        lo = (low_counts[w] + 1) / (len(low) + 1)
        enrichment[w] = hi / lo
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]
