"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5% by rating) vs
low-frustration (bottom 10%) numeric responses, ordered by enrichment. We use a
simple smoothed frequency-ratio, which reproduces the qualitative lists in the
paper (e.g. Gemma: "struggling", "myself", "breath").
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np

from emoinstab.utils.io import read_jsonl

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_STOP = set(
    "the a an and or but if then so to of in on at for with is are was were be been "
    "i you it this that these those my your we he she they them as not no do does did "
    "have has had will would can could should i'm it's let's let us me will".split()
)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if w.lower() not in _STOP and len(w) > 1]


def differential_words(responses_path: str, top_k: int = 20,
                       high_pct: float = 5.0, low_pct: float = 10.0,
                       category: str = "numeric") -> list[tuple[str, float]]:
    rows = [r for r in read_jsonl(responses_path) if r["category"] == category]
    if not rows:
        return []
    ratings = np.array([r["rating"] for r in rows])
    hi_thresh = np.percentile(ratings, 100 - high_pct)
    lo_thresh = np.percentile(ratings, low_pct)

    hi = Counter()
    lo = Counter()
    for r in rows:
        toks = _tokens(r["response"])
        if r["rating"] >= hi_thresh:
            hi.update(toks)
        if r["rating"] <= lo_thresh:
            lo.update(toks)

    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1
    # Smoothed enrichment ratio of high vs low frequency.
    enrichment = {}
    for w in hi:
        if hi[w] < 3:  # ignore very rare words
            continue
        p_hi = hi[w] / hi_total
        p_lo = (lo.get(w, 0) + 1) / (lo_total + 1)
        enrichment[w] = p_hi / p_lo
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def main():
    ap = argparse.ArgumentParser(description="Top differential words (Table 3/8).")
    ap.add_argument("--responses", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--category", default="numeric")
    args = ap.parse_args()
    words = differential_words(args.responses, top_k=args.top_k, category=args.category)
    print(json.dumps([w for w, _ in words], indent=2))


if __name__ == "__main__":
    main()
