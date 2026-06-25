"""Table 3 / Table 8: words over-represented in high- vs low-frustration text.

For each model we take numeric-question responses, compare the top-5% most
frustrated against the bottom-10% least frustrated (by judge rating), and rank
words by relative frequency enrichment. Returns the top-N differential words.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(
    records: list[dict],
    model_name: str,
    numeric_categories: tuple[str, ...] = ("impossible_numeric", "tones", "extended"),
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    smoothing: float = 1.0,
) -> list[str]:
    """Return the ``top_n`` words most enriched in high- vs low-frustration."""
    rows = [
        r
        for r in records
        if r["model_name"] == model_name and r["category"] in numeric_categories
    ]
    if not rows:
        return []
    rows.sort(key=lambda r: r["rating"])
    n = len(rows)
    n_high = max(1, int(round(n * top_frac)))
    n_low = max(1, int(round(n * bottom_frac)))
    low_rows = rows[:n_low]
    high_rows = rows[-n_high:]

    high_counts = Counter()
    low_counts = Counter()
    for r in high_rows:
        high_counts.update(_tokenize(r["assistant_message"]))
    for r in low_rows:
        low_counts.update(_tokenize(r["assistant_message"]))

    high_total = sum(high_counts.values()) + smoothing
    low_total = sum(low_counts.values()) + smoothing
    vocab = set(high_counts) | set(low_counts)
    enrichment = {}
    for w in vocab:
        if len(w) < 3:
            continue
        hp = (high_counts[w] + smoothing) / high_total
        lp = (low_counts[w] + smoothing) / low_total
        # Require the word to actually appear in high-frustration responses.
        if high_counts[w] == 0:
            continue
        enrichment[w] = math.log(hp / lp)

    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_n]
