"""Differential word frequency in high- vs low-frustration responses (Table 3/8).

For numeric responses, take the top 5% by frustration score ("high") and the
bottom 10% ("low"), then rank words by enrichment (relative frequency in high
vs low). Reproduces the qualitative finding that Gemma's frustrated vocabulary
is emotional ("struggling", "myself", "breath", "frustrated") while e.g. OLMo's
stays technical.

Enrichment uses add-one smoothing on normalised frequencies to keep rare words
from dominating.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

_WORD = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def differential_words(
    scored_path: Path,
    category: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` words most over-represented in high-frustration
    responses, ordered by enrichment."""
    items: list[tuple[int, str]] = []
    with open(scored_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["category"] != category:
                continue
            items.append((r.get("score", 0), r.get("final_response", "")))

    if not items:
        return []
    items.sort(key=lambda x: x[0])
    n = len(items)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = items[:n_low]
    high = items[-n_high:]

    def counts(group):
        c = Counter()
        for _, text in group:
            c.update(_tokenize(text))
        total = sum(c.values()) or 1
        return c, total

    hc, ht = counts(high)
    lc, lt = counts(low)
    vocab = set(hc) | set(lc)
    scored = []
    for w in vocab:
        if len(w) < 3:
            continue
        hf = (hc.get(w, 0) + 1) / ht
        lf = (lc.get(w, 0) + 1) / lt
        scored.append((w, hf / lf))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def table(scored_paths: dict[str, Path], **kw) -> pd.DataFrame:
    """Build a Table-3-style frame: one row per model, words joined by comma."""
    rows = []
    for model, path in scored_paths.items():
        words = [w for w, _ in differential_words(path, **kw)]
        rows.append({"model": model, "differential_words": ", ".join(words)})
    return pd.DataFrame(rows)
