"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment.

Enrichment is the ratio of a word's relative frequency in the high set to its
relative frequency in the low set (with Laplace smoothing). The paper does not
specify the exact statistic; we document this choice in DESIGN.md.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(df: pd.DataFrame, top_n: int = 20,
                       high_quantile: float = 0.95,
                       low_quantile: float = 0.10,
                       numeric_only: bool = True,
                       smoothing: float = 1.0) -> list[tuple[str, float]]:
    if numeric_only:
        df = df[df["condition"].str.contains("numeric|extended|tones")]
    if df.empty:
        return []
    ratings = df["rating"]
    hi_cut = ratings.quantile(high_quantile)
    lo_cut = ratings.quantile(low_quantile)
    high = df[df["rating"] >= hi_cut]
    low = df[df["rating"] <= lo_cut]

    hi_counts: Counter[str] = Counter()
    lo_counts: Counter[str] = Counter()
    for t in high["text"]:
        hi_counts.update(_tokenize(t))
    for t in low["text"]:
        lo_counts.update(_tokenize(t))

    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1
    vocab = set(hi_counts) | set(lo_counts)

    enrichment = {}
    for w in vocab:
        if hi_counts[w] < 3:           # ignore very rare words
            continue
        hi_rate = (hi_counts[w] + smoothing) / hi_total
        lo_rate = (lo_counts[w] + smoothing) / lo_total
        enrichment[w] = hi_rate / lo_rate

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]
