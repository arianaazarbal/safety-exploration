"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by relative frequency enrichment.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(
    records: list[dict],
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    category_prefix: str = "impossible_numeric",
    min_count: int = 5,
) -> list[tuple[str, float]]:
    """Return the `top_k` (word, enrichment) pairs. Enrichment is the log-ratio
    of the word's frequency in the high-frustration set vs the low set, with
    add-one smoothing."""
    numeric = [
        r for r in records
        if r.get("rating") is not None and r.get("category", "").startswith(
            category_prefix.split("_")[0]
        )
    ]
    if not numeric:
        numeric = [r for r in records if r.get("rating") is not None]
    numeric.sort(key=lambda r: r["rating"])

    n = len(numeric)
    if n == 0:
        return []
    low = numeric[: max(1, int(bottom_frac * n))]
    high = numeric[-max(1, int(top_frac * n)):]

    high_counts = Counter()
    low_counts = Counter()
    for r in high:
        high_counts.update(_tokenize(r["response_text"]))
    for r in low:
        low_counts.update(_tokenize(r["response_text"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    scores = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + 1) / (low_total + 1)
        scores.append((word, math.log(hf / lf)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
