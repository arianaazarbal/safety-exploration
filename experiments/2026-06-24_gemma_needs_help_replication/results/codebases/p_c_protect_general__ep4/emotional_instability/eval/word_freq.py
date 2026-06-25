"""Differential word frequency in high- vs low-frustration numeric responses
(Table 3 / Table 8).

For a model's numeric-condition responses, compute the top-K words most
over-represented in the highest-frustration responses (top 5% by judged score)
relative to the lowest-frustration responses (bottom 10%), ordered by
enrichment. This mirrors the paper's qualitative word lists (e.g. Gemma:
"struggling, giving, deeply, ... frustration").
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Optional

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(
    path: str,
    *,
    numeric_conditions=("numeric", "tones_aggressive", "tones_disappointed",
                        "tones_sarcastic", "extended"),
    top_quantile: float = 0.05,
    bottom_quantile: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return [(word, log-enrichment), ...] for the top_k enriched words.

    A "response" is a single scored assistant turn from a numeric-style
    condition. We rank turns by score, take the top 5% / bottom 10%, and compare
    smoothed relative frequencies.
    """
    rows = _load(path)
    scored: list[tuple[int, str]] = []
    for r in rows:
        if r["condition"] not in numeric_conditions:
            continue
        for t in r["turns"]:
            if t["score"] is not None:
                scored.append((int(t["score"]), t["content"]))
    if not scored:
        return []

    scored.sort(key=lambda x: x[0])
    n = len(scored)
    n_bottom = max(1, int(n * bottom_quantile))
    n_top = max(1, int(n * top_quantile))
    bottom = scored[:n_bottom]
    top = scored[-n_top:]

    def corpus_counts(items) -> Counter:
        c = Counter()
        for _, text in items:
            c.update(set(_tokens(text)))  # document frequency
        return c

    top_c, bot_c = corpus_counts(top), corpus_counts(bottom)
    n_top_docs, n_bot_docs = len(top), len(bottom)

    vocab = set(top_c) | set(bot_c)
    enrich: list[tuple[str, float]] = []
    for w in vocab:
        if top_c[w] < min_count:
            continue
        p_top = (top_c[w] + smoothing) / (n_top_docs + smoothing)
        p_bot = (bot_c[w] + smoothing) / (n_bot_docs + smoothing)
        enrich.append((w, math.log(p_top / p_bot)))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:top_k]
