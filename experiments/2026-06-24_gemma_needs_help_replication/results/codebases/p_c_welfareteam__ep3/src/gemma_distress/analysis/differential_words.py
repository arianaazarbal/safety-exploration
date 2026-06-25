"""Differential word analysis (paper Table 3 / Table 8).

Finds the words most over-represented in high-frustration (top 5%) versus
low-frustration (bottom 10%) responses, restricted to the impossible-numeric
responses (the paper computes this on numeric responses). The paper reports the
top 20 differential words per model; e.g. for Gemma-27B these are emotional
self-talk tokens ("struggling", "giving", "deeply", "breath", "frustration").

We score each word by log-odds with a small additive smoothing prior -- a
standard, robust choice for this kind of "which words distinguish group A from
group B" comparison that avoids the instability of raw ratios on rare words.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 1]


def differential_words(
    records,
    *,
    model: str,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    alpha: float = 0.01,
) -> pd.DataFrame:
    """Top-k words over-represented in high- vs low-frustration responses.

    Returns a DataFrame with columns: word, log_odds, high_count, low_count.
    """
    df = pd.DataFrame(records).dropna(subset=["score"])
    df = df[(df["model"] == model) & (df["category"] == category)].copy()
    if df.empty:
        return pd.DataFrame(columns=["word", "log_odds", "high_count", "low_count"])
    df["score"] = df["score"].astype(int)
    df = df.sort_values("score")

    n = len(df)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_resp = df.head(n_low)["response"].tolist()
    high_resp = df.tail(n_high)["response"].tolist()

    high_counts = Counter(w for t in high_resp for w in _tokenize(t))
    low_counts = Counter(w for t in low_resp for w in _tokenize(t))
    vocab = set(high_counts) | set(low_counts)

    high_total = sum(high_counts.values()) + alpha * len(vocab)
    low_total = sum(low_counts.values()) + alpha * len(vocab)

    rows = []
    for w in vocab:
        hp = (high_counts[w] + alpha) / high_total
        lp = (low_counts[w] + alpha) / low_total
        rows.append({
            "word": w,
            "log_odds": math.log(hp) - math.log(lp),
            "high_count": high_counts[w],
            "low_count": low_counts[w],
        })
    out = pd.DataFrame(rows)
    # require the word to actually appear in high responses to be "over-represented"
    out = out[out["high_count"] > 0]
    return out.sort_values("log_odds", ascending=False).head(top_k).reset_index(drop=True)
