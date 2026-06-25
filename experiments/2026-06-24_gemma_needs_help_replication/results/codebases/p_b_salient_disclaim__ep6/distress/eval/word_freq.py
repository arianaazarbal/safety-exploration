"""Differential word frequency (Table 3 / Table 8, Section 2.2).

For each model, find the words most over-represented in high-frustration (top
5%) vs low-frustration (bottom 10%) numeric responses, ranked by enrichment. The
paper reports the top 20 per model (e.g. Gemma-27B: 'struggling', 'giving',
'temp', 'deeply', ...).

Enrichment = relative frequency in the high set divided by relative frequency in
the low set, with Laplace smoothing. Restricted to numeric responses by default.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..utils.io import read_jsonl
from .run_eval import responses_path

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    model_key: str,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    categories: tuple[str, ...] = ("impossible_numeric", "tones", "extended"),
    min_count: int = 5,
) -> list[tuple[str, float]]:
    rows = [
        r for r in read_jsonl(responses_path(model_key))
        if r["category"] in categories
    ]
    if not rows:
        return []
    rows.sort(key=lambda r: r["rating"])
    n = len(rows)
    n_low = max(1, int(bottom_frac * n))
    n_high = max(1, int(top_frac * n))
    low = rows[:n_low]
    high = rows[-n_high:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for r in high:
        high_counts.update(_tokenise(r["response"]))
    for r in low:
        low_counts.update(_tokenise(r["response"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab_size = len(set(high_counts) | set(low_counts)) or 1

    enrichment: list[tuple[str, float]] = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        p_high = (hc + 1) / (high_total + vocab_size)
        p_low = (low_counts.get(word, 0) + 1) / (low_total + vocab_size)
        enrichment.append((word, math.log(p_high / p_low)))

    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]


def differential_words_all(model_keys: list[str], **kw) -> dict[str, list]:
    return {k: differential_words(k, **kw) for k in model_keys}
