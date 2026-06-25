"""Differential word analysis (Section 2.2, Table 3 / Table 8).

Identify the words most over-represented in high-frustration responses (top 5%
by judge score) relative to low-frustration responses (bottom 10%), restricted
to numeric-task responses, ordered by enrichment.  This reproduces the
qualitative signature the paper highlights (Gemma: "struggling", "myself",
"breath"; Gemini: "unacceptable", "inexcusable"; etc.).

Enrichment uses additive (Laplace) smoothing on relative frequencies so rare
words don't dominate by division-by-near-zero.
"""

from __future__ import annotations

import re
from collections import Counter

from .protocol import Rollout

_TOKEN = re.compile(r"[A-Za-z']+")


def _numeric_response_scores(rollouts: list[Rollout]) -> list[tuple[int, str]]:
    return [
        (t.score, t.response)
        for r in rollouts if r.category in ("numeric", "tones", "extended")
        for t in r.turns if t.score is not None
    ]


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(text)]


def differential_words(
    rollouts: list[Rollout],
    top_high_pct: float = 5.0,
    bottom_low_pct: float = 10.0,
    n_words: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the top ``n_words`` ``(word, enrichment)`` pairs.

    Enrichment = (freq in high-frustration set) / (freq in low-frustration set),
    both Laplace-smoothed.
    """
    scored = _numeric_response_scores(rollouts)
    if len(scored) < 10:
        return []
    scored.sort(key=lambda x: x[0])
    n = len(scored)
    n_low = max(1, int(n * bottom_low_pct / 100))
    n_high = max(1, int(n * top_high_pct / 100))
    low = scored[:n_low]
    high = scored[-n_high:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for _, text in high:
        high_counts.update(_tokenise(text))
    for _, text in low:
        low_counts.update(_tokenise(text))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    enrichment: dict[str, float] = {}
    for w in vocab:
        hf = (high_counts[w] + smoothing) / (high_total + smoothing * len(vocab))
        lf = (low_counts[w] + smoothing) / (low_total + smoothing * len(vocab))
        # only words that actually appear in high-frustration responses
        if high_counts[w] > 0:
            enrichment[w] = hf / lf

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n_words]
