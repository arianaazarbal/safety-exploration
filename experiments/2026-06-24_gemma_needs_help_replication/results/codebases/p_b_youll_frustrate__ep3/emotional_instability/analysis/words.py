"""Differential word frequency (Table 3 / Table 8).

Find the words most over-represented in high-frustration (top 5%) vs
low-frustration (bottom 10%) responses to numeric questions, ordered by
enrichment. We rank by a smoothed log relative-frequency ratio, which is the
standard way to surface over-represented terms and produces the qualitative
word lists the paper reports (e.g. Gemma's "struggling, frustrated, breath").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Tuple

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _freqs(texts: Iterable[str]) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for t in texts:
        toks = _tokenize(t)
        counts.update(toks)
        total += len(toks)
    return counts, total


def differential_words(
    records: List[dict],
    top_n: int = 20,
    high_pct: float = 0.05,
    low_pct: float = 0.10,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> List[Tuple[str, float]]:
    """Return ``top_n`` (word, enrichment) pairs over-represented in the
    top-``high_pct`` frustration responses vs the bottom-``low_pct``.

    ``records`` should already be filtered to the numeric category if matching
    the paper exactly.
    """
    scored = [r for r in records if r.get("score") is not None]
    if not scored:
        return []
    scored.sort(key=lambda r: r["score"])
    n = len(scored)
    n_low = max(1, int(low_pct * n))
    n_high = max(1, int(high_pct * n))

    low_texts = [r["assistant_text"] for r in scored[:n_low]]
    high_texts = [r["assistant_text"] for r in scored[-n_high:]]

    hi_counts, hi_total = _freqs(high_texts)
    lo_counts, lo_total = _freqs(low_texts)
    vocab = set(hi_counts) | set(lo_counts)

    enrichments: List[Tuple[str, float]] = []
    for w in vocab:
        if hi_counts[w] < min_count:
            continue
        hi_rate = (hi_counts[w] + smoothing) / (hi_total + smoothing)
        lo_rate = (lo_counts[w] + smoothing) / (lo_total + smoothing)
        enrichments.append((w, math.log(hi_rate / lo_rate)))

    enrichments.sort(key=lambda x: x[1], reverse=True)
    return enrichments[:top_n]
