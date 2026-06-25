"""Table 3: words over-represented in high- vs low-frustration numeric responses.

The paper reports, per model, the top-20 words over-represented in the top-5%
(high) vs bottom-10% (low) frustration numeric responses. We reproduce this with
a smoothed frequency-ratio (a standard "differential words" measure): rank words
by the ratio of their relative frequency in the high set vs the low set, with
add-k smoothing to avoid division by zero and a minimum count to suppress noise.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    df,
    model_key: str,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
    smoothing: float = 0.5,
    category: str = "impossible_numeric",
) -> list[tuple[str, float]]:
    """Return the ``top_n`` (word, log-ratio) pairs most over-represented in
    high-frustration numeric responses for ``model_key``.
    """
    sub = df[(df["model_key"] == model_key) & (df["category"] == category)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("frustration_score")
    n = len(sub)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))

    low_texts = sub.head(n_low)["text"].tolist()
    high_texts = sub.tail(n_high)["text"].tolist()

    high_counts: Counter = Counter()
    low_counts: Counter = Counter()
    for t in high_texts:
        high_counts.update(_tokenize(t))
    for t in low_texts:
        low_counts.update(_tokenize(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    scored = []
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        p_high = (high_counts[w] + smoothing) / (high_total + smoothing * len(vocab))
        p_low = (low_counts[w] + smoothing) / (low_total + smoothing * len(vocab))
        scored.append((w, math.log(p_high / p_low)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
