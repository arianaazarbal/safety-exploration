"""Differential word frequency (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses, ordered by relative frequency."

For a given model we take its impossible-numeric responses, split into the top 5%
and bottom 10% by frustration score, compute per-word frequencies in each set, and
rank words by enrichment (high-set rate / low-set rate, with Laplace smoothing).
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

import config

from ..utils import read_jsonl

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(model_name: str, *, top_frac: float = 0.05,
                       bottom_frac: float = 0.10, n: int = 20,
                       category: str = "impossible_numeric",
                       min_count: int = 5) -> pd.DataFrame:
    rows = [
        r for r in read_jsonl(config.RESPONSES_DIR / model_name / f"{category}.jsonl")
        if r["turn"] == r["n_turns"] - 1
    ]
    if not rows:
        return pd.DataFrame(columns=["word", "enrichment", "high_rate", "low_rate"])

    ratings = np.array([r["rating"] for r in rows])
    order = np.argsort(ratings)
    n_total = len(rows)
    low_idx = order[: max(1, int(bottom_frac * n_total))]
    high_idx = order[::-1][: max(1, int(top_frac * n_total))]

    def counts(idxs):
        c = Counter()
        for i in idxs:
            c.update(set(_tokenize(rows[i]["response"])))  # presence, not raw freq
        return c

    high_c, low_c = counts(high_idx), counts(low_idx)
    n_high, n_low = len(high_idx), len(low_idx)

    recs = []
    for word in set(high_c) | set(low_c):
        if high_c[word] + low_c[word] < min_count:
            continue
        high_rate = (high_c[word] + 1) / (n_high + 2)
        low_rate = (low_c[word] + 1) / (n_low + 2)
        recs.append({
            "word": word,
            "enrichment": high_rate / low_rate,
            "high_rate": high_rate,
            "low_rate": low_rate,
        })
    df = pd.DataFrame(recs).sort_values("enrichment", ascending=False)
    return df.head(n).reset_index(drop=True)
