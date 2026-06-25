"""Differential word-frequency analysis (PAPER Table 3 / Table 8).

Find words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ranked by relative frequency enrichment.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..utils import read_jsonl

_WORD_RE = re.compile(r"[a-zA-Z_]+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    jsonl_path: str,
    *,
    category: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 5,
) -> list[tuple[str, float]]:
    """Return the top_k words ranked by (freq_high / freq_low) enrichment."""
    scored = []
    for rec in read_jsonl(jsonl_path):
        if rec["category"] != category or rec.get("final_score") is None:
            continue
        scored.append((rec["final_score"], rec["final_response"] if "final_response" in rec
                       else rec["assistant_turns"][-1]))
    if not scored:
        return []

    scores = np.array([s for s, _ in scored])
    hi_cut = np.quantile(scores, 1 - top_frac)
    lo_cut = np.quantile(scores, bottom_frac)

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    hi_tot = lo_tot = 0
    for score, text in scored:
        toks = _tokenise(text)
        if score >= hi_cut:
            hi_counts.update(toks)
            hi_tot += len(toks)
        elif score <= lo_cut:
            lo_counts.update(toks)
            lo_tot += len(toks)

    if hi_tot == 0 or lo_tot == 0:
        return []

    enrichment = []
    for word, hc in hi_counts.items():
        if hc < min_count:
            continue
        hf = hc / hi_tot
        lf = (lo_counts.get(word, 0) + 1) / (lo_tot + 1)  # Laplace smoothing
        enrichment.append((word, hf / lf))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_k]
