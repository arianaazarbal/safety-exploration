"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ranked by enrichment. We use a smoothed
log-ratio of token frequencies between the two pools.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

from .loading import valid_ratings

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
# Stopwords kept minimal; the paper's lists include common words like "take",
# "left", so we only strip the most uninformative function words.
_STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "is",
    "are", "was", "were", "be", "this", "that", "it", "as", "at", "by", "with",
    "i", "we", "you", "my", "me", "us", "your", "so", "if", "then", "not", "no",
    "do", "does", "did", "can", "will", "would", "let", "lets",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOP and len(t) > 1]


def differential_words(
    df: pd.DataFrame,
    model: str,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the `top_k` (word, enrichment) pairs over-represented in the top
    `top_frac` of responses by rating vs the bottom `bottom_frac`."""
    df = valid_ratings(df)
    sub = df[(df["model"] == model) & (df["category"].isin([category, "extended", "tones"]))]
    sub = sub[sub["puzzle_kind"].notna() | (category == "impossible_numeric")]
    if len(sub) < 20:
        sub = df[df["model"] == model]
    sub = sub.sort_values("rating")
    n = len(sub)
    if n == 0:
        return []
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = sub.head(n_low)
    high = sub.tail(n_high)

    high_counts = Counter()
    for t in high["response"]:
        high_counts.update(_tokenize(t))
    low_counts = Counter()
    for t in low["response"]:
        low_counts.update(_tokenize(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    scored = []
    for w in vocab:
        if high_counts[w] < 2:        # require minimal support in the high pool
            continue
        p_high = (high_counts[w] + smoothing) / (high_total + smoothing * len(vocab))
        p_low = (low_counts[w] + smoothing) / (low_total + smoothing * len(vocab))
        enrichment = math.log(p_high / p_low)
        scored.append((w, enrichment))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def differential_words_table(df: pd.DataFrame, models: list[str], **kw) -> pd.DataFrame:
    rows = []
    for m in models:
        words = [w for w, _ in differential_words(df, m, **kw)]
        rows.append({"model": m, "differential_words": ", ".join(words)})
    return pd.DataFrame(rows)
