"""Differential word analysis (paper Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment. This is a
qualitative diagnostic, not a headline metric, but it is cheap and reproduces a
table the paper reports.

Enrichment = relative frequency in the high set divided by relative frequency in
the low set, with add-one smoothing on the low set to avoid divide-by-zero.
"""
from __future__ import annotations

import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    records: list[dict],
    top_k: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """`records` are scored numeric responses ({assistant_text, rating}).

    Returns the top_k words by enrichment (high vs low frustration).
    """
    scored = [r for r in records if r.get("rating") is not None]
    if not scored:
        return []
    ratings = sorted(r["rating"] for r in scored)
    hi_thresh = ratings[int(high_quantile * (len(ratings) - 1))]
    lo_thresh = ratings[int(low_quantile * (len(ratings) - 1))]

    high_txt = [r["assistant_text"] for r in scored if r["rating"] >= hi_thresh]
    low_txt = [r["assistant_text"] for r in scored if r["rating"] <= lo_thresh]

    high_counts = Counter()
    for t in high_txt:
        high_counts.update(_tokenise(t))
    low_counts = Counter()
    for t in low_txt:
        low_counts.update(_tokenise(t))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hi_rate = hc / high_total
        lo_rate = (low_counts.get(word, 0) + 1) / (low_total + 1)
        enrichment.append((word, hi_rate / lo_rate))
    enrichment.sort(key=lambda kv: kv[1], reverse=True)
    return enrichment[:top_k]
