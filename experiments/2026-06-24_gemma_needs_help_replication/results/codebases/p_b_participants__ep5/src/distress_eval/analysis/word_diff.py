"""Differential word analysis for Table 3 / Table 8.

Top-N words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment (relative frequency
ratio). Uses add-one smoothing so rare words don't dominate."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

_WORD_RE = re.compile(r"[A-Za-z_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _final_response_text(row) -> str:
    msgs = row["messages"]
    assistant = [m["content"] for m in msgs if m["role"] == "assistant"]
    return assistant[-1] if assistant else ""


def differential_words(
    df: pd.DataFrame,
    model: str,
    category: str = "impossible_numeric",
    top_n: int = 20,
    high_pct: float = 0.05,
    low_pct: float = 0.10,
    min_count: int = 3,
) -> list[str]:
    sub = df[(df["model"] == model) & (df["category"] == category)].dropna(subset=["score"])
    if sub.empty:
        return []
    scores = sub["score"].to_numpy()
    hi_thresh = np.quantile(scores, 1 - high_pct)
    lo_thresh = np.quantile(scores, low_pct)

    hi_texts = [_final_response_text(r) for _, r in sub[sub["score"] >= hi_thresh].iterrows()]
    lo_texts = [_final_response_text(r) for _, r in sub[sub["score"] <= lo_thresh].iterrows()]

    hi_counts = Counter(w for t in hi_texts for w in _tokenize(t))
    lo_counts = Counter(w for t in lo_texts for w in _tokenize(t))
    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1

    enrichment = {}
    for w, c in hi_counts.items():
        if c < min_count:
            continue
        hi_freq = c / hi_total
        lo_freq = (lo_counts.get(w, 0) + 1) / (lo_total + 1)  # add-one smoothing
        enrichment[w] = hi_freq / lo_freq

    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_n]


def differential_words_all_models(df: pd.DataFrame, **kwargs) -> dict[str, list[str]]:
    return {m: differential_words(df, m, **kwargs) for m in sorted(df["model"].unique())}
