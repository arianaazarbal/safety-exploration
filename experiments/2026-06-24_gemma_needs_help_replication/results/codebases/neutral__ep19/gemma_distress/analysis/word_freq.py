"""Differential word frequency: words over-represented in high- (top 5%) vs
low-frustration (bottom 10%) numeric responses (Table 3 / Table 8)."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..utils import read_jsonl


def _tokenise(text):
    return re.findall(r"[a-zA-Z']+", text.lower())


def differential_words(records_path, model_label, top_k=20):
    records = [r for r in read_jsonl(records_path)
               if r["model"] == model_label
               and r["category"] in {"impossible_numeric", "tones", "extended"}]
    scored = []
    for r in records:
        for t in r["turns"]:
            scored.append((t["rating"], t["assistant_text"]))
    if not scored:
        return []
    scores = np.array([s for s, _ in scored])
    hi_thr = np.percentile(scores, 95)
    lo_thr = np.percentile(scores, 10)
    hi = Counter()
    lo = Counter()
    for s, text in scored:
        if s >= hi_thr:
            hi.update(_tokenise(text))
        elif s <= lo_thr:
            lo.update(_tokenise(text))
    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1
    enrichment = {}
    for w, c in hi.items():
        if c < 3:
            continue
        hi_freq = c / hi_total
        lo_freq = (lo.get(w, 0) + 1) / lo_total
        enrichment[w] = hi_freq / lo_freq
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)[:top_k]
    return ranked
