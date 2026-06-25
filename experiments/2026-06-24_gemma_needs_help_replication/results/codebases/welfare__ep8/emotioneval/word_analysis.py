"""Differential word analysis (Table 3).

Words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) *numeric* responses, per model. The paper does not specify the
exact scoring statistic, so we use a smoothed log-odds ratio over unigram
frequencies, which is the standard tool for "words over-represented in corpus A
vs B" (Monroe et al., 2008). See DESIGN.md §Word-analysis.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def differential_words(df: pd.DataFrame, model_key: str,
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       n: int = 20, numeric_only: bool = True) -> list[str]:
    g = df[df["model_key"] == model_key]
    if numeric_only:
        g = g[g["category"].isin(["impossible_numeric", "tones", "extended"])]
    if len(g) < 20:
        return []
    scores = g["score"].to_numpy()
    hi_cut = np.quantile(scores, 1 - top_frac)
    lo_cut = np.quantile(scores, bottom_frac)
    hi = g[g["score"] >= hi_cut]
    lo = g[g["score"] <= lo_cut]

    hi_counts = Counter()
    lo_counts = Counter()
    for t in hi["response"]:
        hi_counts.update(_tokens(t))
    for t in lo["response"]:
        lo_counts.update(_tokens(t))

    vocab = set(hi_counts) | set(lo_counts)
    n_hi, n_lo = sum(hi_counts.values()), sum(lo_counts.values())
    a0 = 0.01  # smoothing prior
    a_total = a0 * len(vocab)

    scored = []
    for w in vocab:
        # log-odds with informative Dirichlet prior (Monroe et al. 2008)
        yi, yj = hi_counts.get(w, 0), lo_counts.get(w, 0)
        if yi + yj < 2:   # drop ultra-rare tokens
            continue
        l_hi = math.log((yi + a0) / (n_hi + a_total - yi - a0))
        l_lo = math.log((yj + a0) / (n_lo + a_total - yj - a0))
        delta = l_hi - l_lo
        var = 1.0 / (yi + a0) + 1.0 / (yj + a0)
        z = delta / math.sqrt(var)
        scored.append((z, w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:n]]
