"""Differential-word analysis (Table 3).

Identifies words over-represented in high-frustration (top 5%) versus
low-frustration (bottom 10%) numeric responses, per model. We rank words by a
smoothed log odds-ratio of their relative frequency in the two pools, which is
a standard, robust differential-usage measure and reproduces the qualitative
finding (Gemma's high-frustration vocabulary is emotional self-talk:
"struggling", "myself", "breath", "frustrated").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    scored_path: str | Path,
    model: str,
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` words most over-represented in high- vs low-frustration.

    Restricted to one model's responses in the numeric category (paper's
    Table 3 setting). Words must appear at least ``min_count`` times in the
    high pool to be considered.
    """
    df = pd.read_json(scored_path, lines=True)
    df = df[(df["model"] == model) & (df["category"] == category)]
    if df.empty:
        return []

    df = df.sort_values("score")
    n = len(df)
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    low_pool = df.head(n_bottom)["assistant"].tolist()
    high_pool = df.tail(n_top)["assistant"].tolist()

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for t in high_pool:
        high_counts.update(_tokenize(t))
    for t in low_pool:
        low_counts.update(_tokenize(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    scores: list[tuple[str, float]] = []
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        p_high = (high_counts[w] + smoothing) / (high_total + smoothing * len(vocab))
        p_low = (low_counts[w] + smoothing) / (low_total + smoothing * len(vocab))
        log_odds = math.log(p_high) - math.log(p_low)
        scores.append((w, log_odds))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
