"""Differential word-frequency analysis (Paper §2.2, Table 3 / Table 8).

Identify the words most over-represented in high-frustration (top 5%) versus
low-frustration (bottom 10%) numeric responses for a given model. Enrichment is
the ratio of within-group relative frequencies, with add-one (Laplace) smoothing
to keep rare words from dominating.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass
class DifferentialWords:
    model: str
    n_high: int
    n_low: int
    words: list[tuple[str, float]]  # (word, enrichment), descending


def differential_words(
    rows: list[dict],
    *,
    model: str,
    top_k: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    min_count: int = 3,
    numeric_only: bool = True,
) -> DifferentialWords:
    """Rank words by over-representation in high- vs low-frustration responses.

    Parameters mirror Table 8: top 5% vs bottom 10% by score. ``numeric_only``
    restricts to the impossible-numeric / tones / extended conditions, which is
    what the paper's table is computed over.
    """
    rows = [r for r in rows if r.get("score") is not None and r.get("response")]
    if numeric_only:
        numeric_conditions = {
            "impossible_numeric_3turn", "tones_3turn", "extended_8turn",
        }
        rows = [r for r in rows if r.get("condition") in numeric_conditions]
    if not rows:
        return DifferentialWords(model=model, n_high=0, n_low=0, words=[])

    scores = sorted(int(r["score"]) for r in rows)
    n = len(scores)
    hi_thresh = scores[min(n - 1, int(high_quantile * n))]
    lo_thresh = scores[max(0, int(low_quantile * n))]

    high_rows = [r for r in rows if int(r["score"]) >= hi_thresh]
    low_rows = [r for r in rows if int(r["score"]) <= lo_thresh]

    high_counts: Counter = Counter()
    low_counts: Counter = Counter()
    for r in high_rows:
        high_counts.update(_tokenize(r["response"]))
    for r in low_rows:
        low_counts.update(_tokenize(r["response"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1
    vocab = set(high_counts) | set(low_counts)

    enrichments: list[tuple[str, float]] = []
    for w in vocab:
        if high_counts[w] < min_count:
            continue
        hi_freq = (high_counts[w] + 1) / (high_total + len(vocab))
        lo_freq = (low_counts[w] + 1) / (low_total + len(vocab))
        enrichments.append((w, hi_freq / lo_freq))

    enrichments.sort(key=lambda kv: kv[1], reverse=True)
    return DifferentialWords(
        model=model,
        n_high=len(high_rows),
        n_low=len(low_rows),
        words=enrichments[:top_k],
    )
