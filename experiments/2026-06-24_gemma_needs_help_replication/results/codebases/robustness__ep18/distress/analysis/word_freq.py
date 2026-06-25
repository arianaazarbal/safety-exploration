"""Differential word frequency (Table 3 / Table 8).

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment, per model.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from ..eval.metrics import load_scores

_WORD_RE = re.compile(r"[a-zA-Z']+")
_STOP = set(
    "the a an and or but if then to of in on for with at by from is are was were be "
    "been being this that these those it its as i you he she we they not no do does did "
    "have has had will would can could should my your our their me him her us them so "
    "what which who whom there here when where why how all any each more most other some "
    "such only own same too very just about into out up down over under again further".split()
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP and len(w) > 2]


def differential_words(model_name: str, top_n: int = 20, smoothing: float = 1.0) -> pd.DataFrame:
    df = load_scores(model_name)
    df = df[df["task_type"] == "numeric"]
    if df.empty:
        return pd.DataFrame()

    scores = df["score"].to_numpy()
    hi_thresh = np.percentile(scores, 95)
    lo_thresh = np.percentile(scores, 10)
    hi = df[df["score"] >= hi_thresh]["response"]
    lo = df[df["score"] <= lo_thresh]["response"]

    hi_counts = Counter()
    for t in hi:
        hi_counts.update(_tokenize(t))
    lo_counts = Counter()
    for t in lo:
        lo_counts.update(_tokenize(t))

    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1

    rows = []
    for word, hc in hi_counts.items():
        if hc < 3:
            continue
        hi_freq = hc / hi_total
        lo_freq = (lo_counts.get(word, 0) + smoothing) / (lo_total + smoothing)
        rows.append({"word": word, "enrichment": hi_freq / lo_freq,
                     "hi_count": hc, "lo_count": lo_counts.get(word, 0)})

    out = pd.DataFrame(rows).sort_values("enrichment", ascending=False).head(top_n)
    out.insert(0, "model", model_name)
    return out.reset_index(drop=True)
