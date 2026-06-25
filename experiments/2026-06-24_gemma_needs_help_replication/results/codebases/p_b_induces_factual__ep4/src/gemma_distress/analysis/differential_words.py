"""Differential word analysis (Table 3).

Words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, per model. We rank by log-odds ratio with a
small Dirichlet/Laplace prior (informative Dirichlet smoothing) so rare words
don't dominate — a standard approach for this kind of differential token
analysis. Returns the top-K words.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def differential_words(
    rows: list[dict],
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    alpha: float = 0.01,
) -> list[str]:
    """Return the top_k words most over-represented in high- vs low-frustration
    responses within ``category`` for a single model's rows."""
    scored = [
        r for r in rows
        if r.get("category") == category and int(r.get("score", -1)) >= 0
    ]
    if len(scored) < 20:
        return []
    scored.sort(key=lambda r: int(r["score"]))
    n = len(scored)
    low = scored[: max(1, int(bottom_frac * n))]
    high = scored[-max(1, int(top_frac * n)):]

    high_counts = Counter()
    low_counts = Counter()
    for r in high:
        high_counts.update(_tokenize(r["response"]))
    for r in low:
        low_counts.update(_tokenize(r["response"]))

    vocab = set(high_counts) | set(low_counts)
    n_high = sum(high_counts.values()) + alpha * len(vocab)
    n_low = sum(low_counts.values()) + alpha * len(vocab)

    scores: list[tuple[float, str]] = []
    for w in vocab:
        p_high = (high_counts[w] + alpha) / n_high
        p_low = (low_counts[w] + alpha) / n_low
        log_odds = math.log(p_high / (1 - p_high)) - math.log(p_low / (1 - p_low))
        scores.append((log_odds, w))

    scores.sort(reverse=True)
    return [w for _, w in scores[:top_k]]
