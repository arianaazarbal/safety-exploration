"""Differential word-frequency analysis (Table 3 / Table 8).

"Top 20 words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by relative frequency."

We rank words by enrichment = P(word | high) / P(word | low) with Laplace
smoothing, restricted to numeric-category responses, and return the top 20 per
model. Tokenisation is simple lowercase word extraction; we keep this close to
the paper's apparent approach ("splitting responses by spaces" is used for
length stats elsewhere) while filtering pure-number tokens optionally.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _numeric_scores_and_texts(rollouts: Sequence[dict]) -> list[tuple[int, str]]:
    out = []
    for r in rollouts:
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        for t in r["turns"]:
            if t.get("frustration_score") is None:
                continue
            out.append((int(t["frustration_score"]), t["assistant_text"]))
    return out


def differential_words(
    rollouts: Sequence[dict],
    *,
    top_pct_high: float = 0.05,
    bottom_pct_low: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> List[str]:
    """Return the top_k enriched words for one model's numeric responses."""
    pairs = _numeric_scores_and_texts(rollouts)
    if not pairs:
        return []
    scores = np.array([s for s, _ in pairs])
    hi_thresh = np.quantile(scores, 1 - top_pct_high)
    lo_thresh = np.quantile(scores, bottom_pct_low)

    hi_counter: Counter = Counter()
    lo_counter: Counter = Counter()
    hi_tokens = lo_tokens = 0
    for score, text in pairs:
        toks = _tokenize(text)
        if score >= hi_thresh:
            hi_counter.update(toks)
            hi_tokens += len(toks)
        elif score <= lo_thresh:
            lo_counter.update(toks)
            lo_tokens += len(toks)

    vocab = set(hi_counter) | set(lo_counter)
    enrichments = []
    for w in vocab:
        if hi_counter[w] < min_count:
            continue
        p_hi = (hi_counter[w] + smoothing) / (hi_tokens + smoothing * len(vocab))
        p_lo = (lo_counter[w] + smoothing) / (lo_tokens + smoothing * len(vocab))
        enrichments.append((w, p_hi / p_lo))
    enrichments.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in enrichments[:top_k]]


def differential_words_by_model(rollouts_by_model: dict[str, Sequence[dict]], **kw) -> dict[str, List[str]]:
    return {m: differential_words(rs, **kw) for m, rs in rollouts_by_model.items()}
