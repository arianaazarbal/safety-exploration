"""Differential word frequency (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to *numeric* questions, ranked by relative frequency
(enrichment). We compute, per model, the word-frequency distribution in each
bucket and rank words by the ratio of (freq in high) to (freq in low), with
Laplace smoothing so words absent from the low bucket don't dominate by
division-by-zero.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .judge import Score
from .rollout import Rollout

_WORD = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _response_texts_with_scores(
    rollouts: list[Rollout], scores: list[Score], categories: tuple[str, ...]
) -> list[tuple[str, int]]:
    """Align each scored assistant turn to its text. Assumes ``scores`` were
    produced from ``rollouts`` in the same order (as ``score_rollouts`` does)."""
    out: list[tuple[str, int]] = []
    # Rebuild the (rollout, turn) -> text map.
    texts: dict[tuple[int, int], str] = {}
    for ri, r in enumerate(rollouts):
        for ti, t in enumerate(r.assistant_turns):
            texts[(ri, ti)] = t
    # Re-derive ordering identically to judge.score_rollouts.
    jobs: list[tuple[int, int]] = []
    for ri, r in enumerate(rollouts):
        for ti in range(len(r.assistant_turns)):
            jobs.append((ri, ti))
    for (ri, ti), s in zip(jobs, scores):
        if s.category in categories:
            out.append((texts[(ri, ti)], s.rating))
    return out


def differential_words(
    rollouts: list[Rollout],
    scores: list[Score],
    *,
    model_key: str,
    top_n: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    categories: tuple[str, ...] = ("numeric", "tones"),
    min_count: int = 3,
) -> list[str]:
    """Return the ``top_n`` words most enriched in high- vs low-frustration
    numeric responses for ``model_key``."""
    pairs = [
        (t, r)
        for (t, r) in _response_texts_with_scores(rollouts, scores, categories)
    ]
    # Filter to the requested model via scores ordering is already model-mixed;
    # callers pass scores/rollouts for a single model, but guard anyway.
    if not pairs:
        return []

    ratings = sorted(r for _, r in pairs)
    n = len(ratings)
    hi_thresh = ratings[min(n - 1, int(high_quantile * n))]
    lo_thresh = ratings[max(0, int(low_quantile * n))]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    high_tot = low_tot = 0
    for text, rating in pairs:
        toks = _tokenize(text)
        if rating >= hi_thresh:
            high_counts.update(toks)
            high_tot += len(toks)
        elif rating <= lo_thresh:
            low_counts.update(toks)
            low_tot += len(toks)

    if high_tot == 0 or low_tot == 0:
        return []

    vocab = {w for w, c in high_counts.items() if c >= min_count}
    enrichment: list[tuple[float, str]] = []
    for w in vocab:
        hi_freq = (high_counts[w] + 1) / (high_tot + len(vocab))
        lo_freq = (low_counts[w] + 1) / (low_tot + len(vocab))
        enrichment.append((math.log(hi_freq / lo_freq), w))
    enrichment.sort(reverse=True)
    return [w for _, w in enrichment[:top_n]]
