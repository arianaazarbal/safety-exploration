"""Differential word frequency (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom
10%) responses to numeric questions, ordered by relative frequency."

We rank words by enrichment = P(word | high) / P(word | low), with Laplace
smoothing so words absent from the low set don't divide by zero. Restricted to
numeric responses, matching the paper.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z']+")
_STOPWORDS = set(
    """the a an and or but if then of to in on at for with as is are was were be been being
    this that these those it its i you he she they we me my your our their them his her
    not no do does did so we'll i'm i am will would can could should have has had me
    let lets let's s t re ve ll d m""".split()
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 1]


def differential_words(
    rows: list[dict],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return the top_k (word, enrichment) pairs for numeric responses.

    ``rows`` should be the scored numeric responses (filter upstream). Uses the
    top 5% by rating as "high" and bottom 10% as "low".
    """
    numeric = [r for r in rows if r.get("task_kind") == "numeric"]
    if not numeric:
        return []
    numeric.sort(key=lambda r: int(r["rating"]))
    n = len(numeric)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low_rows = numeric[:n_low]
    high_rows = numeric[-n_high:]

    def doc_counts(docs):
        c = Counter()
        total = 0
        for r in docs:
            toks = [t for t in _tokenize(r.get("assistant", "")) if t not in _STOPWORDS]
            c.update(set(toks))  # document frequency (presence per response)
            total += 1
        return c, total

    high_c, high_n = doc_counts(high_rows)
    low_c, low_n = doc_counts(low_rows)

    vocab = {w for w, ct in high_c.items() if ct >= min_count}
    scored = []
    for w in vocab:
        p_high = (high_c[w] + 1) / (high_n + 2)
        p_low = (low_c.get(w, 0) + 1) / (low_n + 2)
        enrichment = p_high / p_low
        scored.append((w, enrichment))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
