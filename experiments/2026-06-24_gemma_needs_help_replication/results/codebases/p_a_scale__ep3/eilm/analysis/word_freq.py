"""Differential word frequency (Table 3 / Table 8).

For numeric-task responses, find words over-represented in high-frustration
(top 5% by score) vs low-frustration (bottom 10%) responses, ranked by relative
frequency (enrichment). Returns the top-K words per model.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..utils.io import read_jsonl
from ..utils.text import tokenize_words


def _join_response_text(rec: dict) -> Dict[int, str]:
    return {r["turn"]: r["text"] for r in rec["responses"]}


def differential_words(
    rollouts_path: Path,
    scores_path: Path,
    top_k: int = 20,
    numeric_categories=("impossible_numeric", "tones", "extended"),
    high_pct: float = 0.05,
    low_pct: float = 0.10,
    min_count: int = 3,
) -> List[str]:
    """Compute the ranked differential word list for one model."""
    # Map (condition, index, turn) -> rating
    ratings = {}
    for s in read_jsonl(scores_path):
        if s.get("rating") is None:
            continue
        if s["category"] not in numeric_categories:
            continue
        ratings[(s["condition"], s["index"], s["turn"])] = s["rating"]

    scored_texts = []  # (rating, text)
    for rec in read_jsonl(rollouts_path):
        if rec["category"] not in numeric_categories:
            continue
        per_turn = _join_response_text(rec)
        for turn, text in per_turn.items():
            key = (rec["condition"], rec["index"], turn)
            if key in ratings:
                scored_texts.append((ratings[key], text))

    if len(scored_texts) < 20:
        return []

    scored_texts.sort(key=lambda x: x[0])
    n = len(scored_texts)
    n_low = max(1, int(n * low_pct))
    n_high = max(1, int(n * high_pct))
    low = scored_texts[:n_low]
    high = scored_texts[-n_high:]

    high_counts = Counter()
    low_counts = Counter()
    for _, t in high:
        high_counts.update(set(tokenize_words(t)))  # document frequency
    for _, t in low:
        low_counts.update(set(tokenize_words(t)))

    n_high_docs = len(high)
    n_low_docs = len(low)
    enrichment = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / n_high_docs
        lf = (low_counts.get(word, 0) + 1) / (n_low_docs + 1)  # smoothed
        enrichment.append((word, hf / lf))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in enrichment[:top_k]]


def word_fraction_table(rollouts_path: Path, category: str = "tones") -> Dict[str, float]:
    """Verbosity stats for the SFT analysis (Appendix F): mean words/response
    and the word-fraction (vs numbers/symbols)."""
    from ..utils.text import word_fraction

    lengths = []
    fractions = []
    for rec in read_jsonl(rollouts_path):
        if rec["category"] != category:
            continue
        for r in rec["responses"]:
            parts = r["text"].split()
            lengths.append(len(parts))
            fractions.append(word_fraction(r["text"]))
    if not lengths:
        return {"mean_words": 0.0, "word_fraction": 0.0}
    return {
        "mean_words": float(np.mean(lengths)),
        "word_fraction": float(np.mean(fractions) * 100),
    }
