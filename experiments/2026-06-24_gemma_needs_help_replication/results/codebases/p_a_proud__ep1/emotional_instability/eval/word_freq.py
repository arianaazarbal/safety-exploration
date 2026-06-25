"""Table 3 / Table 8: words over-represented in high- vs low-frustration numeric
responses.

For a model, take its impossible-numeric responses (final turns), rank by judge
score, take the top 5% (high) and bottom 10% (low) buckets, and rank words by the
ratio of their frequency in the high bucket to the low bucket (with smoothing).
Returns the top-N enriched words.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from ..config import SCORED_DIR
from .schema import read_jsonl

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def differential_words(
    model_key: str,
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
    scored_dir=SCORED_DIR,
) -> pd.DataFrame:
    """Top ``top_n`` words enriched in high- vs low-frustration responses."""
    texts, scores = [], []
    for c in read_jsonl(scored_dir / f"{model_key}.jsonl"):
        if c.category != category:
            continue
        ft = c.final_turn
        if ft.score is None:
            continue
        texts.append(ft.assistant)
        scores.append(ft.score)
    if not texts:
        return pd.DataFrame(columns=["word", "enrichment", "high_freq", "low_freq"])

    order = np.argsort(scores)  # ascending
    n = len(texts)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_idx = order[:n_low]
    high_idx = order[-n_high:]

    high_counts: Counter[str] = Counter()
    low_counts: Counter[str] = Counter()
    for i in high_idx:
        high_counts.update(_tokenize(texts[i]))
    for i in low_idx:
        low_counts.update(_tokenize(texts[i]))

    high_total = max(1, sum(high_counts.values()))
    low_total = max(1, sum(low_counts.values()))
    eps = 1.0 / low_total  # Laplace-ish smoothing for words absent from low bucket

    rows = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = low_counts.get(word, 0) / low_total
        enrichment = hf / (lf + eps)
        rows.append((word, enrichment, hf, lf))

    out = pd.DataFrame(rows, columns=["word", "enrichment", "high_freq", "low_freq"])
    return out.sort_values("enrichment", ascending=False).head(top_n).reset_index(drop=True)
