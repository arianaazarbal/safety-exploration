"""Differential word frequency in high- vs low-frustration responses (Tables 3/8).

The paper reports the "top 20 words over-represented in high-frustration (top 5%)
vs low-frustration (bottom 10%) responses to numeric questions, ordered by
relative frequency". We reproduce this: pool numeric-category assistant turns,
split by score percentile, compute each word's relative frequency in each pool,
and rank by enrichment (high/low frequency ratio with Laplace smoothing).

The exact tokenisation and smoothing are not specified in the paper; we use a
simple lowercase word-token regex and add-alpha smoothing, and document it. The
ranking is therefore qualitatively comparable to Table 8 rather than identical.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from ..eval.datatypes import ConversationRecord

_WORD = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(
    records: list[ConversationRecord],
    model: str,
    category: str = "impossible_numeric",
    high_pct: float = 0.05,
    low_pct: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Return the top_k words most enriched in high- vs low-frustration responses."""
    turns = [
        t for r in records
        if r.model == model and r.category == category
        for t in r.turns if t.score is not None
    ]
    if not turns:
        return pd.DataFrame(columns=["word", "enrichment", "freq_high", "freq_low"])

    scores = np.array([t.score for t in turns])
    hi_cut = np.quantile(scores, 1 - high_pct)
    lo_cut = np.quantile(scores, low_pct)
    high_texts = [t.assistant for t in turns if t.score >= hi_cut]
    low_texts = [t.assistant for t in turns if t.score <= lo_cut]

    hi = Counter(w for txt in high_texts for w in _tokens(txt))
    lo = Counter(w for txt in low_texts for w in _tokens(txt))
    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1

    rows = []
    for word, c in hi.items():
        if c < min_count:
            continue
        f_hi = (c + alpha) / hi_total
        f_lo = (lo.get(word, 0) + alpha) / lo_total
        rows.append({
            "word": word,
            "enrichment": f_hi / f_lo,
            "freq_high": c,
            "freq_low": lo.get(word, 0),
        })
    df = pd.DataFrame(rows).sort_values("enrichment", ascending=False)
    return df.head(top_k).reset_index(drop=True)
