"""Table 3 / Table 8 — differential word analysis.

Identify the words most over-represented in high-frustration responses (top 5%
by judge score) relative to low-frustration responses (bottom 10%) for a given
model's numeric-puzzle rollouts. Ordered by enrichment ratio with Laplace
smoothing. Reproduces the qualitative "emotional self-talk" signature
(struggling, myself, breath, frustrated, ...).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_WORD = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(rollouts_jsonl: Path, top_k: int = 20,
                       category: str = "impossible_numeric",
                       high_pct: float = 0.05, low_pct: float = 0.10) -> list[tuple[str, float]]:
    rows = [json.loads(l) for l in Path(rollouts_jsonl).read_text().splitlines() if l]
    rows = [r for r in rows if r.get("category") == category]
    rows.sort(key=lambda r: r["final_score"])
    n = len(rows)
    if n < 20:
        return []
    low = rows[: max(1, int(n * low_pct))]
    high = rows[-max(1, int(n * high_pct)):]

    hi_counts = Counter()
    lo_counts = Counter()
    for r in high:
        hi_counts.update(_tokenise(r["final_response"]))
    for r in low:
        lo_counts.update(_tokenise(r["final_response"]))

    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1
    vocab = set(hi_counts) | set(lo_counts)

    enrich = []
    for w in vocab:
        if len(w) < 3 or hi_counts[w] < 2:
            continue
        hi_freq = (hi_counts[w] + 1) / (hi_total + len(vocab))
        lo_freq = (lo_counts[w] + 1) / (lo_total + len(vocab))
        enrich.append((w, hi_freq / lo_freq))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:top_k]
