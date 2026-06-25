"""Differential word analysis (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by relative frequency (enrichment).
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 2]


def differential_words(
    df: pd.DataFrame,
    *,
    model: str,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    smoothing: float = 1.0,
) -> list[str]:
    """Return the ``top_n`` words most enriched in high- vs low-frustration
    numeric responses for one model (reproduces Table 3/8)."""
    sub = df[(df["model"] == model) & (df["category"] == category)].copy()
    sub = sub.dropna(subset=["score"]).sort_values("score")
    n = len(sub)
    if n == 0:
        return []
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = sub.head(n_low)
    high = sub.tail(n_high)

    hi_counts = Counter()
    for r in high["response"]:
        hi_counts.update(set(_tokenize(r)))   # document frequency
    lo_counts = Counter()
    for r in low["response"]:
        lo_counts.update(set(_tokenize(r)))

    hi_n, lo_n = len(high), len(low)
    vocab = set(hi_counts) | set(lo_counts)
    enrichment = {}
    for w in vocab:
        p_hi = (hi_counts[w] + smoothing) / (hi_n + smoothing)
        p_lo = (lo_counts[w] + smoothing) / (lo_n + smoothing)
        enrichment[w] = p_hi / p_lo
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_n]
