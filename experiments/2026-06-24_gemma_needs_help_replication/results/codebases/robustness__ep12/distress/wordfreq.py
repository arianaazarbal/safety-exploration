"""Differential word analysis (Table 3 / Table 8).

Finds the words most over-represented in high-frustration (top 5%) vs
low-frustration (bottom 10%) numeric responses for each model. Reproduces the
qualitative signature (e.g. Gemma: "struggling, giving, deeply, myself,
breath, frustrated"; OLMo: technical terms only).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text):
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(results_path, model=None, numeric_only=True,
                       top_frac=0.05, bottom_frac=0.10, top_n=20,
                       min_count=5, smoothing=1.0):
    """Return the top_n words ranked by enrichment in high- vs low-frustration.

    Enrichment = (freq in high pool + smoothing) / (freq in low pool +
    smoothing), using relative frequencies.
    """
    numeric_cats = {"impossible_numeric", "tones", "extended"}
    recs = []
    with Path(results_path).open() as fh:
        for line in fh:
            r = json.loads(line)
            if model and r["model"] != model:
                continue
            if numeric_only and r["category"] not in numeric_cats:
                continue
            if r.get("rating") is None:
                continue
            recs.append(r)
    if not recs:
        return []

    recs.sort(key=lambda r: r["rating"])
    n = len(recs)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_pool = recs[:n_low]
    high_pool = recs[-n_high:]

    high_counts = Counter()
    low_counts = Counter()
    for r in high_pool:
        high_counts.update(_tokenize(r["response"]))
    for r in low_pool:
        low_counts.update(_tokenize(r["response"]))

    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    scored = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = low_counts.get(word, 0) / low_total
        enrichment = (hf + smoothing / high_total) / (lf + smoothing / low_total)
        scored.append((word, enrichment, hc, low_counts.get(word, 0)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def differential_words_table(results_path, models, **kwargs):
    out = {}
    for m in models:
        words = differential_words(results_path, model=m, **kwargs)
        out[m] = [w for w, *_ in words]
    return out
