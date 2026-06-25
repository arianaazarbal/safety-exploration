"""Differential word analysis (Table 3).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses, computed per model on the impossible-numeric responses.

We score words by log-odds ratio with a Dirichlet (informative) prior, following
Monroe et al. (2008) "Fightin' Words", which is the standard robust method for
this kind of over-representation ranking (the paper does not specify its exact
statistic; see DESIGN.md).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(text)]


def _counts(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(_tokenize(t))
    return c


def differential_words(
    df: pd.DataFrame,
    model: str,
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Return the top_k words most over-represented in high- vs low-frustration."""
    sub = df[(df["model"] == model) & (df["category"] == category)].copy()
    if sub.empty:
        raise ValueError(f"No rows for model={model}, category={category}")
    sub = sub.sort_values("frustration")
    n = len(sub)
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    low = sub.head(n_bottom)["response"].tolist()
    high = sub.tail(n_top)["response"].tolist()

    c_high = _counts(high)
    c_low = _counts(low)
    vocab = set(c_high) | set(c_low)
    n_high = sum(c_high.values())
    n_low = sum(c_low.values())
    a0 = alpha * len(vocab)

    rows = []
    for w in vocab:
        yi = c_high.get(w, 0)
        yj = c_low.get(w, 0)
        # log-odds with informative Dirichlet prior + variance
        log_odds = np.log((yi + alpha) / (n_high + a0 - yi - alpha)) - np.log(
            (yj + alpha) / (n_low + a0 - yj - alpha)
        )
        var = 1.0 / (yi + alpha) + 1.0 / (yj + alpha)
        z = log_odds / np.sqrt(var)
        rows.append((w, yi, yj, log_odds, z))

    out = pd.DataFrame(rows, columns=["word", "n_high", "n_low", "log_odds", "z"])
    out = out.sort_values("z", ascending=False).head(top_k).reset_index(drop=True)
    return out
