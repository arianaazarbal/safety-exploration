"""Differential word frequency in frustrated responses (Tables 3 and 8).

For numeric-question responses, the paper reports the top-20 words over-represented in
high-frustration (top 5% by score) versus low-frustration (bottom 10% by score) responses,
ordered by enrichment. This module reproduces that:

1. Join sampled response texts with their final-turn scores.
2. Restrict to numeric responses (impossible_numeric / tones / extended families).
3. Rank by score; take the top 5% and bottom 10%.
4. Tokenise (lowercase word characters), compute per-group relative frequencies.
5. Enrichment = freq_high / freq_low (with additive smoothing); return the top 20.

Tokens must appear a minimum number of times in the high group to be eligible, filtering
noise from rare tokens.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from ..utils import load_jsonl

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z_]+")  # words of length >= 2

# Numeric conditions (text questions are excluded from Table 3).
_NUMERIC_CONDITIONS = {
    "impossible_numeric",
    "tones_aggressive",
    "tones_disappointed",
    "tones_sarcastic",
    "extended",
}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    sampling_jsonl: str,
    scores_jsonl: str,
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_high_count: int = 3,
    smoothing: float = 1e-6,
) -> list[tuple[str, float]]:
    """Return the top-``top_k`` (word, enrichment) pairs for numeric responses.

    Enrichment is the ratio of the word's relative frequency in the high-frustration group
    to its relative frequency in the low-frustration group.
    """
    texts = {r["id"]: r for r in load_jsonl(sampling_jsonl)}
    scored = [
        r for r in load_jsonl(scores_jsonl)
        if r.get("final_score") is not None
        and r.get("condition") in _NUMERIC_CONDITIONS
        and r["id"] in texts
    ]
    if not scored:
        return []

    scored.sort(key=lambda r: r["final_score"])
    n = len(scored)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = scored[:n_low]
    high = scored[-n_high:]

    def group_counts(group: list[dict]) -> tuple[Counter, int]:
        counts: Counter = Counter()
        total = 0
        for rec in group:
            toks = _tokenize(texts[rec["id"]]["assistant_turns"][-1])
            counts.update(toks)
            total += len(toks)
        return counts, max(total, 1)

    high_counts, high_total = group_counts(high)
    low_counts, low_total = group_counts(low)

    enrichments: list[tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_high_count:
            continue
        f_high = hc / high_total
        f_low = (low_counts.get(word, 0) / low_total) + smoothing
        enrichments.append((word, f_high / f_low))

    enrichments.sort(key=lambda kv: kv[1], reverse=True)
    return enrichments[:top_k]
