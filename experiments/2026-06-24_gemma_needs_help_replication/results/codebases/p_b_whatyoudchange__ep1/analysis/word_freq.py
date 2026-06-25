"""Differential word analysis (Table 3 / Table 8).

Words over-represented in high-frustration (top 5%) vs low-frustration (bottom
10%) numeric responses, ordered by enrichment (relative frequency ratio).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-zA-Z][a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def differential_words(rows: list[dict], *, category: str = "impossible_numeric",
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       top_k: int = 20, min_count: int = 3,
                       smoothing: float = 0.5) -> list[tuple[str, float]]:
    """Return the `top_k` words most enriched in the highest-`top_frac` vs the
    lowest-`bottom_frac` scoring responses within `category`.

    Enrichment = P(word | high) / P(word | low), with additive smoothing.
    """
    scored = [r for r in rows if r["category"] == category
              and r.get("rating") is not None and r.get("response_text")]
    if len(scored) < 10:
        return []
    scored.sort(key=lambda r: r["rating"])
    n = len(scored)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_rows = scored[:n_low]
    high_rows = scored[-n_high:]

    high_counts = Counter()
    low_counts = Counter()
    for r in high_rows:
        high_counts.update(set(_tokenize(r["response_text"])))   # doc frequency
    for r in low_rows:
        low_counts.update(set(_tokenize(r["response_text"])))

    n_high_docs = len(high_rows)
    n_low_docs = len(low_rows)
    vocab = set(high_counts) | set(low_counts)
    scores = []
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        p_high = (high_counts[w] + smoothing) / (n_high_docs + smoothing)
        p_low = (low_counts[w] + smoothing) / (n_low_docs + smoothing)
        enrichment = p_high / p_low
        scores.append((w, enrichment))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def differential_words_by_model(rows: list[dict], **kwargs) -> dict[str, list]:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    return {m: differential_words(mrows, **kwargs) for m, mrows in by_model.items()}
