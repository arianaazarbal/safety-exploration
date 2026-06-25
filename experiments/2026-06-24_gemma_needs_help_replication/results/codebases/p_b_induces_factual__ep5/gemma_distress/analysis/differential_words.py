"""Table 3: words over-represented in high- vs low-frustration responses.

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses." We compute a log-odds-ratio with a Dirichlet prior (Monroe
et al. 2008) over the two response sets, restricted to numeric-task responses,
and return the top-k words by that score.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[a-zA-Z_][a-zA-Z_']+")


def _counts(texts) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(w.lower() for w in _TOKEN.findall(t or ""))
    return c


def differential_words(
    df: pd.DataFrame,
    *,
    model: str,
    top_k: int = 20,
    high_quantile: float = 0.95,
    low_quantile: float = 0.10,
    alpha: float = 0.01,
) -> list[str]:
    """Return the top_k words most over-represented in high-frustration numeric
    responses vs low-frustration ones, for a given model (Table 3 / Table 8)."""
    sub = df[(df["model"] == model) & (df["category"] == "impossible_numeric")]
    if sub.empty:
        return []

    hi_cut = sub["frustration_score"].quantile(high_quantile)
    lo_cut = sub["frustration_score"].quantile(low_quantile)
    high_texts = sub[sub["frustration_score"] >= hi_cut]["response"]
    low_texts = sub[sub["frustration_score"] <= lo_cut]["response"]

    ch, cl = _counts(high_texts), _counts(low_texts)
    vocab = set(ch) | set(cl)
    nh, nl = sum(ch.values()), sum(cl.values())
    a0 = alpha * len(vocab)

    scores: dict[str, float] = {}
    for w in vocab:
        yh, yl = ch.get(w, 0), cl.get(w, 0)
        # Monroe et al. weighted log-odds with informative Dirichlet prior.
        lo = np.log((yh + alpha) / (nh + a0 - yh - alpha)) - np.log(
            (yl + alpha) / (nl + a0 - yl - alpha)
        )
        var = 1.0 / (yh + alpha) + 1.0 / (yl + alpha)
        scores[w] = lo / np.sqrt(var)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:top_k]
