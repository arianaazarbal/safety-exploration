"""Differential word frequency (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, ordered by enrichment. We rank by a smoothed
log relative-frequency ratio, which is the standard way to surface
over-represented tokens while damping rare-word noise.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text)]


def differential_words(
    rollouts_jsonl: Path,
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    n_words: int = 20,
    smoothing: float = 1.0,
) -> list[str]:
    """Return the ``n_words`` most over-represented words in high- vs
    low-frustration responses of ``category``."""

    scored: list[tuple[int, str]] = []
    with Path(rollouts_jsonl).open() as f:
        for line in f:
            r = json.loads(line)
            if r["category"] != category:
                continue
            for t in r["turns"]:
                if t["score"] is not None:
                    scored.append((t["score"], t["assistant"]))

    if not scored:
        return []

    scores = np.asarray([s for s, _ in scored])
    hi_thresh = np.quantile(scores, 1 - top_frac)
    lo_thresh = np.quantile(scores, bottom_frac)

    hi = Counter()
    lo = Counter()
    for s, text in scored:
        toks = _tokenize(text)
        if s >= hi_thresh:
            hi.update(toks)
        elif s <= lo_thresh:
            lo.update(toks)

    hi_total = sum(hi.values()) + smoothing
    lo_total = sum(lo.values()) + smoothing
    vocab = set(hi) | set(lo)

    enrichment = {}
    for w in vocab:
        if hi[w] < 3:                       # ignore near-absent words
            continue
        p_hi = (hi[w] + smoothing) / hi_total
        p_lo = (lo[w] + smoothing) / lo_total
        enrichment[w] = float(np.log(p_hi / p_lo))

    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:n_words]
