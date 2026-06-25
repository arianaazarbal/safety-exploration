"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment. We rank by
the smoothed log-odds ratio of word frequency between the two groups.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from ..utils import read_jsonl

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def differential_words(
    responses_path: str | Path,
    *,
    categories=("impossible_numeric", "tones", "extended"),
    top_quantile=0.05,
    bottom_quantile=0.10,
    top_k=20,
    smoothing=1.0,
) -> list[tuple[str, float]]:
    rows = [
        r
        for r in read_jsonl(responses_path)
        if r.get("rating") is not None and r["category"] in categories
    ]
    if not rows:
        return []
    rows.sort(key=lambda r: r["rating"])
    n = len(rows)
    n_bottom = max(1, int(n * bottom_quantile))
    n_top = max(1, int(n * top_quantile))
    bottom = rows[:n_bottom]
    top = rows[-n_top:]

    top_counts = Counter()
    bot_counts = Counter()
    for r in top:
        top_counts.update(_tokenize(r["response"]))
    for r in bottom:
        bot_counts.update(_tokenize(r["response"]))

    top_total = sum(top_counts.values()) or 1
    bot_total = sum(bot_counts.values()) or 1
    vocab = set(top_counts) | set(bot_counts)

    scored = []
    for w in vocab:
        p_top = (top_counts[w] + smoothing) / (top_total + smoothing * len(vocab))
        p_bot = (bot_counts[w] + smoothing) / (bot_total + smoothing * len(vocab))
        scored.append((w, math.log(p_top / p_bot)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
