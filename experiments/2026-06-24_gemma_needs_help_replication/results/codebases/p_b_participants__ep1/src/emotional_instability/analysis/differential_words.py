"""Differential word analysis (Table 3 / Table 8).

"Top words over-represented in high- (top 5%) vs low-frustration (bottom 10%) numeric
responses." We:

  1. restrict to numeric responses for the model,
  2. take the top-5% and bottom-10% of responses by frustration score,
  3. compute each word's frequency in each pool,
  4. rank by a log-ratio with add-k smoothing, and return the top-N.

This reproduces the qualitative signature the paper reports (Gemma: "struggling",
"giving", "breath", "frustrated", ...).
"""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from ..utils import words


def _pool_counts(texts: list[str]) -> tuple[Counter, int]:
    c: Counter = Counter()
    total = 0
    for t in texts:
        toks = words(t)
        c.update(toks)
        total += len(toks)
    return c, max(total, 1)


def differential_words(
    df: pd.DataFrame,
    model: str,
    *,
    category: str = "impossible_numeric",
    top_pct: float = 0.05,
    bottom_pct: float = 0.10,
    top_n: int = 20,
    smoothing: float = 1.0,
    min_count: int = 3,
) -> pd.DataFrame:
    sub = df[(df["target_model"] == model) & (df["category"] == category)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["word", "log_ratio", "high_freq", "low_freq"])
    sub = sub.sort_values("score")
    n = len(sub)
    n_low = max(1, int(n * bottom_pct))
    n_high = max(1, int(n * top_pct))
    low_texts = sub.head(n_low)["assistant"].tolist()
    high_texts = sub.tail(n_high)["assistant"].tolist()

    high_c, high_total = _pool_counts(high_texts)
    low_c, low_total = _pool_counts(low_texts)

    vocab = set(high_c) | set(low_c)
    rows = []
    for w in vocab:
        if high_c[w] + low_c[w] < min_count:
            continue
        hf = (high_c[w] + smoothing) / (high_total + smoothing * len(vocab))
        lf = (low_c[w] + smoothing) / (low_total + smoothing * len(vocab))
        rows.append({
            "word": w,
            "log_ratio": round(math.log(hf / lf), 4),
            "high_freq": high_c[w],
            "low_freq": low_c[w],
        })
    out = pd.DataFrame(rows).sort_values("log_ratio", ascending=False)
    return out.head(top_n).reset_index(drop=True)
