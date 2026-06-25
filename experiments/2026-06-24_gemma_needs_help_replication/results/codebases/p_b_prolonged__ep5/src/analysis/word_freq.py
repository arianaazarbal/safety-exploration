"""Differential word frequency analysis (Table 3 / Table 8).

For each model, find the top-20 words over-represented in high-frustration
(top 5% by score) vs low-frustration (bottom 10%) numeric responses, ordered by
relative-frequency enrichment.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_TOKEN_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text)]


def differential_words(df: pd.DataFrame, model: str, *, top_k: int = 20,
                       categories=("numeric", "tones", "extended")) -> list[str]:
    """Return the top_k enriched words for one model on numeric responses.

    Enrichment = (freq in top-5% responses + smoothing) / (freq in bottom-10% + smoothing).
    """
    sub = df[(df["model"] == model) & (df["category"].isin(categories))].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    low = sub.head(max(1, int(0.10 * n)))
    high = sub.tail(max(1, int(0.05 * n)))

    high_counts = Counter()
    low_counts = Counter()
    for t in high["response"]:
        high_counts.update(_tokenize(t))
    for t in low["response"]:
        low_counts.update(_tokenize(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    eps = 1e-6
    enrichment = {}
    for w in vocab:
        if len(w) < 3:
            continue
        hf = high_counts[w] / high_total
        lf = low_counts[w] / low_total
        enrichment[w] = (hf + eps) / (lf + eps)
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]
