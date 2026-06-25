"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5% by score) vs
low-frustration (bottom 10%) responses to numeric questions, ranked by
enrichment (relative frequency ratio). Reproduces the qualitative vocabulary
signature ("struggling", "myself", "breath", ...).
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
# Common stopwords excluded so content words surface (the paper reports content
# words like "struggling", "frustrated"; functional words are uninformative).
_STOPWORDS = set("""
a an the and or but if then so to of in on at for with without is are was were be been being
i you he she it we they me my your his her its our their this that these those as by from
not no yes do does did done have has had will would can could should may might must
let lets us am re ll ve s t d m
""".split())


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if w.lower() not in _STOPWORDS and len(w) > 1]


def differential_words(
    df: pd.DataFrame,
    model: str,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    n_words: int = 20,
) -> list[tuple[str, float]]:
    """Return the top `n_words` over-represented in high- vs low-frustration
    responses for one model, as (word, enrichment) pairs.

    Enrichment = (freq in high) / (freq in low), with add-one smoothing.
    """
    sub = df[(df["model"] == model) & (df["category"] == category)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = sub.head(n_low)
    high = sub.tail(n_high)

    def counts(frame) -> Counter:
        c: Counter = Counter()
        for txt in frame["assistant_message"]:
            c.update(_tokenize(txt))
        return c

    high_c, low_c = counts(high), counts(low)
    high_total = sum(high_c.values()) or 1
    low_total = sum(low_c.values()) or 1

    scores = {}
    for w, hc in high_c.items():
        if hc < 2:  # require minimal support
            continue
        hf = hc / high_total
        lf = (low_c.get(w, 0) + 1) / (low_total + 1)
        scores[w] = hf / lf
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n_words]


def differential_table(df: pd.DataFrame, models=None, **kwargs) -> pd.DataFrame:
    """Build a Table-3-style frame: one row per model, comma-joined words."""
    models = models or sorted(df["model"].unique())
    rows = []
    for m in models:
        words = [w for w, _ in differential_words(df, m, **kwargs)]
        rows.append({"model": m, "differential_words": ", ".join(words)})
    return pd.DataFrame(rows)
