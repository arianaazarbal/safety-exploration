"""Differential word frequency (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment.

We rank by a smoothed log relative-frequency ratio between the two groups,
restricted to numeric-category responses (as in the paper).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(
    df: pd.DataFrame,
    *,
    numeric_categories=("impossible_numeric", "tones", "extended"),
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[str]:
    """Return the ``top_k`` words most enriched in high- vs low-frustration
    numeric responses for a single model's score dataframe."""
    sub = df[df["category"].isin(numeric_categories)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))

    low = sub.head(n_low)
    high = sub.tail(n_high)

    high_counts = Counter()
    low_counts = Counter()
    for t in high["response"]:
        high_counts.update(_tokenize(t))
    for t in low["response"]:
        low_counts.update(_tokenize(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment = {}
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        # Laplace-smoothed relative frequency ratio.
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + 1) / (low_total + 1)
        enrichment[word] = np.log(hf / lf)

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]
