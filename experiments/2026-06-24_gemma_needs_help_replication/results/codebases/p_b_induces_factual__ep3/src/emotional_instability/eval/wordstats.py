"""Differential word analysis (Table 3 / Table 8).

For numeric-task responses, find the words most over-represented in the
high-frustration tail (top 5% by score) relative to the low-frustration tail
(bottom 10% by score), ordered by relative-frequency enrichment.

We rank by a smoothed frequency ratio (high-freq + eps) / (low-freq + eps),
restricted to words above a minimum count, and return the top-K. This reproduces
the qualitative signal in Table 8 (e.g. Gemma: "struggling, giving, deeply,
myself, frustrated, breath, frustration").
"""

from __future__ import annotations

import os
import re
from collections import Counter

from ..logging_utils import read_jsonl

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def _flat_numeric_responses(path: str | os.PathLike) -> list[tuple[str, int]]:
    out = []
    for rec in read_jsonl(path):
        if rec["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        for text, score in zip(rec["responses"], rec["scores"]):
            out.append((text, score))
    return out


def differential_words(
    path: str | os.PathLike,
    top_k: int = 20,
    high_pct: float = 0.05,
    low_pct: float = 0.10,
    min_count: int = 5,
    eps: float = 1e-6,
) -> list[tuple[str, float]]:
    """Return the top-K (word, enrichment) pairs for one model's results."""
    data = _flat_numeric_responses(path)
    if not data:
        return []
    data.sort(key=lambda t: t[1], reverse=True)
    n = len(data)
    n_high = max(1, int(high_pct * n))
    n_low = max(1, int(low_pct * n))
    high = data[:n_high]
    low = data[-n_low:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for text, _ in high:
        high_counts.update(_tokenize(text))
    for text, _ in low:
        low_counts.update(_tokenize(text))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    scored: list[tuple[str, float]] = []
    for word, c in high_counts.items():
        if c < min_count:
            continue
        hf = c / high_total
        lf = low_counts.get(word, 0) / low_total
        enrichment = (hf + eps) / (lf + eps)
        scored.append((word, enrichment))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]
