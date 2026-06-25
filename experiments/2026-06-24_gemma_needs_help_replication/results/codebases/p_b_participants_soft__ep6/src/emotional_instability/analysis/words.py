"""Differential word analysis -> Table 3.

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses." We rank words by a smoothed log-odds ratio of their relative
frequency in the high-frustration set vs the low-frustration set. Restricted to
numeric-category responses (impossible_numeric, tones, extended), as in the paper.
"""

from __future__ import annotations

import math
import re
from collections import Counter

NUMERIC_CATEGORIES = {"impossible_numeric", "tones", "extended"}
_WORD = re.compile(r"[A-Za-z']+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(rollouts: list[dict], top_n: int = 20, smoothing: float = 0.5) -> list[tuple[str, float]]:
    """Return the ``top_n`` (word, log_odds) pairs over-represented in high vs low."""
    scored: list[tuple[int, str]] = []
    for roll in rollouts:
        if roll["category"] not in NUMERIC_CATEGORIES:
            continue
        final = roll["turns"][-1]
        if final["frustration"] is not None:
            scored.append((final["frustration"], final["response"]))
    if not scored:
        return []

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    n_low = max(1, int(round(0.10 * n)))  # bottom 10%
    n_high = max(1, int(round(0.05 * n)))  # top 5%
    low = scored[:n_low]
    high = scored[-n_high:]

    high_counts = Counter(w for _, t in high for w in _tokens(t))
    low_counts = Counter(w for _, t in low for w in _tokens(t))
    vocab = set(high_counts) | set(low_counts)
    h_total = sum(high_counts.values()) + smoothing * len(vocab)
    l_total = sum(low_counts.values()) + smoothing * len(vocab)

    ranked = []
    for w in vocab:
        p_high = (high_counts[w] + smoothing) / h_total
        p_low = (low_counts[w] + smoothing) / l_total
        # Require the word to actually appear in the high set.
        if high_counts[w] == 0:
            continue
        ranked.append((w, math.log(p_high / p_low)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
