"""Differential word frequency (Table 3 / Table 8).

For each model, find words over-represented in high-frustration (top 5% by score)
vs low-frustration (bottom 10%) numeric responses, ranked by enrichment.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"[a-zA-Z']+")


def _counts(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for w in _TOKEN.findall((t or "").lower()):
            if len(w) > 2:
                c[w] += 1
    return c


def differential_words(df: pd.DataFrame, model: str, top_k: int = 20,
                       category: str = "impossible_numeric",
                       high_q: float = 0.95, low_q: float = 0.10,
                       smoothing: float = 1.0) -> list[tuple[str, float]]:
    sub = df[(df["model"] == model)]
    if category:
        sub = sub[sub["category"] == category]
    if sub.empty:
        return []
    hi_cut = sub["frustration"].quantile(high_q)
    lo_cut = sub["frustration"].quantile(low_q)
    high = sub[sub["frustration"] >= hi_cut]["response"].tolist()
    low = sub[sub["frustration"] <= lo_cut]["response"].tolist()

    hc, lc = _counts(high), _counts(low)
    h_total = sum(hc.values()) + smoothing
    l_total = sum(lc.values()) + smoothing

    enrich = {}
    for w in hc:
        if hc[w] < 2:  # ignore singletons
            continue
        h_rate = (hc[w] + smoothing) / h_total
        l_rate = (lc.get(w, 0) + smoothing) / l_total
        enrich[w] = float(np.log(h_rate / l_rate))
    ranked = sorted(enrich.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]
