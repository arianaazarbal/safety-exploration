"""Differential word analysis (Table 3 / Table 8): words over-represented in
high-frustration (top 5%) vs low-frustration (bottom 10%) responses to numeric
questions, ordered by enrichment."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(
    df: pd.DataFrame,
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    sub = df[df["category"].isin(_NUMERIC_CATEGORIES)].copy()
    sub = sub.sort_values("rating")
    n = len(sub)
    if n < 10:
        return []
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = sub.head(n_low)
    high = sub.tail(n_high)

    high_counts = Counter()
    for t in high["assistant_text"]:
        high_counts.update(set(_tokenize(t)))     # document frequency
    low_counts = Counter()
    for t in low["assistant_text"]:
        low_counts.update(set(_tokenize(t)))

    high_n, low_n = len(high), len(low)
    scored = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        high_rate = (hc + smoothing) / (high_n + smoothing)
        low_rate = (low_counts.get(word, 0) + smoothing) / (low_n + smoothing)
        scored.append((word, high_rate / low_rate))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
