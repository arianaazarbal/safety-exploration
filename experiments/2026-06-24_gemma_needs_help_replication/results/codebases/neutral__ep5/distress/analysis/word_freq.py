"""Differential word frequency: high- vs low-frustration responses (Table 3/8).

For each model, take the top-5% highest-scoring and bottom-10% lowest-scoring
numeric responses, and rank words by enrichment (relative frequency ratio).
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    df: pd.DataFrame,            # tidy frame from eval.metrics.rollouts_to_dataframe + response text
    response_lookup: dict,       # (model, idx) -> response text; see note below
    *,
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    smoothing: float = 1.0,
) -> dict[str, list[str]]:
    """Return {model: [top_k differential words]} for numeric responses.

    ``df`` must contain columns: model, category, rating, and a ``response``
    column with the raw text (Section 2 runner can be extended to emit it; the
    convenience wrapper ``from_rollouts`` below handles this).
    """
    out: dict[str, list[str]] = {}
    numeric = df[(df["category"] == "impossible_numeric") & df["rating"].notna()]
    for model, g in numeric.groupby("model"):
        g = g.sort_values("rating")
        n = len(g)
        if n < 10:
            out[model] = []
            continue
        low = g.head(max(1, int(n * bottom_frac)))
        high = g.tail(max(1, int(n * top_frac)))
        out[model] = _rank(high["response"].tolist(), low["response"].tolist(), top_k, smoothing)
    return out


def _rank(high_texts, low_texts, top_k, smoothing) -> list[str]:
    hi = Counter()
    lo = Counter()
    for t in high_texts:
        hi.update(_tokenise(t))
    for t in low_texts:
        lo.update(_tokenise(t))
    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1

    scored = []
    for w, c in hi.items():
        if c < 2:
            continue
        hi_rate = c / hi_total
        lo_rate = (lo.get(w, 0) + smoothing) / (lo_total + smoothing)
        scored.append((w, hi_rate / lo_rate))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in scored[:top_k]]


def from_rollouts(rollouts) -> pd.DataFrame:
    """Build a frame with a ``response`` text column for word-frequency analysis."""
    rows = []
    for r in rollouts:
        for t in r.turns:
            rows.append({
                "model": r.model_key, "category": r.category,
                "rating": t.rating, "response": t.response,
            })
    return pd.DataFrame(rows)
