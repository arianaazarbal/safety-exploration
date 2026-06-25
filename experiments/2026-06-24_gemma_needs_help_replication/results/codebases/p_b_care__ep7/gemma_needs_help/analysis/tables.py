"""Differential-word analysis (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ranked by relative frequency enrichment.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..io_utils import read_jsonl

_WORD = re.compile(r"[a-zA-Z][a-zA-Z']+")
NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(
    judged_path: str,
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the top_n (word, enrichment) pairs for one model's judged file."""
    rows = [r for r in read_jsonl(judged_path) if r["category"] in NUMERIC_CATEGORIES]
    if not rows:
        return []
    scores = np.array([r["score"] for r in rows])
    order = np.argsort(scores)
    n = len(rows)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_idx = order[:n_low]
    high_idx = order[-n_high:]

    high_counts: Counter = Counter()
    low_counts: Counter = Counter()
    for i in high_idx:
        high_counts.update(_tokenize(rows[i]["response"]))
    for i in low_idx:
        low_counts.update(_tokenize(rows[i]["response"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment = {}
    for word, hc in high_counts.items():
        if hc < 3:  # ignore very rare words
            continue
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + smoothing) / (low_total + smoothing)
        enrichment[word] = hf / lf
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]
