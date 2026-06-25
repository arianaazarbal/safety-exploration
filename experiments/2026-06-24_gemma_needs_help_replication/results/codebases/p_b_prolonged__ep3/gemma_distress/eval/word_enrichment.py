"""Differential word analysis (Table 3 / Table 8).

"Top 20 words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
numeric responses, ordered by relative frequency."

We compute, per model, the words whose frequency in the top-5%-scored numeric
responses most exceeds their frequency in the bottom-10%-scored numeric
responses, using a smoothed log relative-frequency ratio.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from .. import config
from ..utils.io import read_jsonl

_WORD_RE = re.compile(r"[A-Za-z_]+")


def _numeric_responses_with_scores(model_name: str):
    path = config.RESULTS_DIR / "elicitation" / model_name / "impossible_numeric.jsonl"
    for rec in read_jsonl(path):
        for t in rec["turns"]:
            yield t["assistant_text"], t["rating"]


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    model_name: str,
    top_q: float = 0.95,
    bottom_q: float = 0.10,
    top_k: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Return the ``top_k`` words ranked by enrichment in high- vs low-frustration
    numeric responses."""
    rows = list(_numeric_responses_with_scores(model_name))
    if not rows:
        return []
    scores = sorted(r for _, r in rows)
    n = len(scores)
    hi_thresh = scores[min(n - 1, int(math.ceil(top_q * n)) - 1)]
    lo_thresh = scores[max(0, int(math.floor(bottom_q * n)) - 1)]

    hi = Counter()
    lo = Counter()
    hi_total = lo_total = 0
    for text, score in rows:
        toks = _tokenize(text)
        if score >= hi_thresh:
            hi.update(toks)
            hi_total += len(toks)
        elif score <= lo_thresh:
            lo.update(toks)
            lo_total += len(toks)

    if hi_total == 0 or lo_total == 0:
        return []

    vocab = set(hi) | set(lo)
    enrichment = {}
    for w in vocab:
        p_hi = (hi[w] + smoothing) / (hi_total + smoothing * len(vocab))
        p_lo = (lo[w] + smoothing) / (lo_total + smoothing * len(vocab))
        enrichment[w] = math.log(p_hi / p_lo)

    # Only keep words that actually appear in the high group.
    ranked = sorted(
        ((w, e) for w, e in enrichment.items() if hi[w] > 0),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked[:top_k]
