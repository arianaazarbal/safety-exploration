"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5% by score) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment. We use a smoothed log relative-
frequency ratio, which is robust to rare tokens and matches the paper's "ordered by
relative frequency / enrichment" description.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .store import JsonlStore

_WORD = re.compile(r"[a-zA-Z][a-zA-Z'_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _join_response_text(store: JsonlStore, rollout_id: str, turn_index: int,
                        rollout_index: dict) -> str:
    rec = rollout_index.get(rollout_id)
    if not rec:
        return ""
    for t in rec.get("turns", []):
        if t["turn_index"] == turn_index:
            return t.get("assistant_text", "")
    return ""


def differential_words(
    store: JsonlStore,
    model: str,
    *,
    rollouts_kind: str = "rollouts",
    scores_kind: str = "scores",
    categories=("impossible_numeric", "extended", "tones"),
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 5,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the top_k enriched words for `model` over numeric responses."""
    rollout_index = {r["task_id"]: r for r in store.iter_records(rollouts_kind)}
    scores = [
        s for s in store.iter_records(scores_kind)
        if s.get("model") == model and s.get("rating", -1) >= 0
        and s.get("category") in categories
    ]
    if not scores:
        return []
    scores.sort(key=lambda s: s["rating"])
    n = len(scores)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = scores[:n_low]
    high = scores[-n_high:]

    def counts(subset) -> Counter:
        c: Counter = Counter()
        for s in subset:
            text = _join_response_text(store, s["rollout_id"], s["turn_index"], rollout_index)
            c.update(set(_tokenize(text)))  # document frequency (presence), not raw count
        return c

    hi_c, lo_c = counts(high), counts(low)
    hi_docs, lo_docs = len(high), len(low)

    import math
    scored: list[tuple[str, float]] = []
    vocab = set(hi_c) | set(lo_c)
    for w in vocab:
        if hi_c[w] + lo_c[w] < min_count:
            continue
        p_hi = (hi_c[w] + smoothing) / (hi_docs + 2 * smoothing)
        p_lo = (lo_c[w] + smoothing) / (lo_docs + 2 * smoothing)
        scored.append((w, math.log(p_hi / p_lo)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def differential_table(store: JsonlStore, models: list[str], **kw) -> pd.DataFrame:
    rows = []
    for m in models:
        words = [w for w, _ in differential_words(store, m, **kw)]
        rows.append({"model": m, "differential_words": ", ".join(words)})
    return pd.DataFrame(rows)
