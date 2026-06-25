"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high- (top 5%) vs low- (bottom 10%) frustration numeric
responses, ranked by relative-frequency enrichment.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..utils.io import read_jsonl

NUMERIC_CATEGORIES = {"impossible_numeric", "extended", "tones"}
_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_STOP = set(
    "the a an and or but of to in on for with is are was were be been being this that "
    "it its as at by from we you i he she they them me my your our their not no can will "
    "would should could do does did have has had if then so just about into out up down "
    "over under again more most some any all one two".split()
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if len(w) > 2 and w.lower() not in _STOP]


def differential_words(output_dir: str | Path, model: str, top_n: int = 20,
                       high_pct: float = 0.05, low_pct: float = 0.10) -> list[str]:
    mdir = Path(output_dir) / "section2" / model
    responses = []  # (score, text)
    for path in sorted(mdir.glob("*.jsonl")):
        for roll in read_jsonl(path):
            if roll["category"] not in NUMERIC_CATEGORIES:
                continue
            for t in roll["turns"]:
                if t.get("judged_score") is not None:
                    responses.append((float(t["judged_score"]), t["assistant"]))
    if not responses:
        return []

    scores = np.asarray([r[0] for r in responses])
    hi_thresh = np.quantile(scores, 1 - high_pct)
    lo_thresh = np.quantile(scores, low_pct)
    high = [t for s, t in responses if s >= hi_thresh]
    low = [t for s, t in responses if s <= lo_thresh]

    hi_counts, lo_counts = Counter(), Counter()
    for t in high:
        hi_counts.update(set(_tokenize(t)))  # document frequency
    for t in low:
        lo_counts.update(set(_tokenize(t)))

    n_hi, n_lo = max(1, len(high)), max(1, len(low))
    vocab = set(hi_counts) | set(lo_counts)
    enrichment = {}
    for w in vocab:
        f_hi = (hi_counts[w] + 1) / (n_hi + 2)
        f_lo = (lo_counts[w] + 1) / (n_lo + 2)
        if hi_counts[w] >= 2:  # ignore singletons
            enrichment[w] = f_hi / f_lo
    return [w for w, _ in sorted(enrichment.items(), key=lambda kv: -kv[1])[:top_n]]
