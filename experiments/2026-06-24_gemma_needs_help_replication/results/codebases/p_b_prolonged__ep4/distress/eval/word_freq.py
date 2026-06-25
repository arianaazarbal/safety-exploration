"""Differential word frequency (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses, ordered by relative frequency."

We restrict to numeric-category responses (as the paper does), rank responses by
frustration rating, take the top 5% and bottom 10%, tokenise into lowercased word
tokens, and score each word by enrichment:

    enrichment(w) = freq_high(w) / (freq_low(w) + eps)

with add-one smoothing on the low-frustration side. Ties in rating are broken by
turn (later turns first) so the 5%/10% cuts are deterministic. Returns the top-20
words per model.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd

_WORD_RE = re.compile(r"[a-zA-Z_]+")


def _tokenise(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _word_freqs(responses: list[str]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for r in responses:
        toks = _tokenise(r)
        counts.update(toks)
        total += len(toks)
    return counts, total


def differential_words(
    judged_rows: list[dict],
    model: str | None = None,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    eps: float = 1.0,
) -> list[tuple[str, float]]:
    """Top-k enriched words for one model's numeric responses."""
    df = pd.DataFrame([r for r in judged_rows if r["rating"] >= 0 and r["category"] == "numeric"])
    if model is not None:
        df = df[df["model"] == model]
    if df.empty:
        return []
    df = df.sort_values(["rating", "turn"], ascending=[False, False]).reset_index(drop=True)
    n = len(df)
    n_high = max(1, int(round(n * top_frac)))
    n_low = max(1, int(round(n * bottom_frac)))
    high = df.head(n_high)["response"].tolist()
    low = df.tail(n_low)["response"].tolist()

    hc, ht = _word_freqs(high)
    lc, lt = _word_freqs(low)

    scores: dict[str, float] = {}
    for w, c in hc.items():
        if c < 2:  # ignore hapax noise
            continue
        f_high = c / ht
        f_low = (lc.get(w, 0) + eps) / (lt + eps)
        scores[w] = f_high / f_low
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def differential_words_all_models(judged_rows: list[dict], **kw) -> dict[str, list[str]]:
    models = sorted({r["model"] for r in judged_rows})
    return {m: [w for w, _ in differential_words(judged_rows, model=m, **kw)] for m in models}
