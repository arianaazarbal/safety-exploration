"""Differential word frequency (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment. We use a smoothed
log-ratio of word probabilities in each group.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_]+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text)]


def _counts(texts) -> Counter:
    c = Counter()
    for t in texts:
        c.update(_tokenise(t))
    return c


def differential_words(df: pd.DataFrame, model: str, *, top_n: int = 20,
                       high_q: float = 0.95, low_q: float = 0.10,
                       smoothing: float = 1.0,
                       min_count: int = 3) -> list[str]:
    """Words enriched in high- vs low-frustration numeric responses for `model`.

    high_q=0.95 -> top 5%; low_q=0.10 -> bottom 10% (Table 8 definition).
    """
    sub = df[(df["model"] == model) &
             (df["category"] == "impossible-numeric")].copy()
    if sub.empty:
        return []
    hi_thresh = sub["rating"].quantile(high_q)
    lo_thresh = sub["rating"].quantile(low_q)
    high_texts = sub[sub["rating"] >= hi_thresh]["assistant_text"]
    low_texts = sub[sub["rating"] <= lo_thresh]["assistant_text"]

    hi = _counts(high_texts)
    lo = _counts(low_texts)
    hi_total = sum(hi.values()) + smoothing
    lo_total = sum(lo.values()) + smoothing

    scores = {}
    for word, c in hi.items():
        if c < min_count:
            continue
        p_hi = (c + smoothing) / hi_total
        p_lo = (lo.get(word, 0) + smoothing) / lo_total
        scores[word] = np.log(p_hi / p_lo)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]
