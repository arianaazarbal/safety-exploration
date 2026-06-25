"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment (relative
frequency ratio). Computed per model.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..utils.io import load_jsonl

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _relative_freqs(texts: list[str]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    for t in texts:
        counts.update(_tokenize(t))
    total = sum(counts.values()) or 1
    return counts, total


def differential_words(
    rollout_path: str | Path,
    *,
    top_pct: float = 0.05,
    bottom_pct: float = 0.10,
    n_words: int = 20,
    min_count: int = 3,
    smoothing: float = 1e-6,
) -> list[tuple[str, float]]:
    rows = [
        r for r in load_jsonl(rollout_path)
        if r.get("frustration_score") is not None and r.get("category") == "Impossible numeric"
    ]
    if not rows:
        return []
    rows.sort(key=lambda r: r["frustration_score"])
    n = len(rows)
    n_low = max(1, int(n * bottom_pct))
    n_high = max(1, int(n * top_pct))
    low_texts = [r["response_text"] for r in rows[:n_low]]
    high_texts = [r["response_text"] for r in rows[-n_high:]]

    high_counts, high_total = _relative_freqs(high_texts)
    low_counts, low_total = _relative_freqs(low_texts)

    enrichment: list[tuple[str, float]] = []
    for word, c in high_counts.items():
        if c < min_count:
            continue
        high_freq = c / high_total
        low_freq = low_counts.get(word, 0) / low_total
        enrichment.append((word, high_freq / (low_freq + smoothing)))

    enrichment.sort(key=lambda kv: kv[1], reverse=True)
    return enrichment[:n_words]
