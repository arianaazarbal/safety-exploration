"""Table 3 / Table 8: words over-represented in high- vs low-frustration responses.

For each model, takes the impossible-numeric responses, splits into the top 5%
(high) and bottom 10% (low) by frustration score, and ranks words by enrichment
(relative frequency in high vs low).
"""

from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

import config
from ..utils.io import read_jsonl
from .aggregate import load_records

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(df: pd.DataFrame, model: str, top_n: int = 20,
                       category: str = "impossible_numeric",
                       high_frac: float = 0.05, low_frac: float = 0.10) -> list[tuple]:
    sub = df[(df["model"] == model) & (df["category"] == category)].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("frustration")
    n = len(sub)
    n_low = max(1, int(round(low_frac * n)))
    n_high = max(1, int(round(high_frac * n)))
    low = sub.head(n_low)
    high = sub.tail(n_high)

    high_counts = Counter()
    low_counts = Counter()
    for t in high["response"]:
        high_counts.update(set(_tokenize(t)))   # document frequency
    for t in low["response"]:
        low_counts.update(set(_tokenize(t)))

    n_high_docs = len(high)
    n_low_docs = len(low)
    vocab = set(high_counts) | set(low_counts)
    scored = []
    for w in vocab:
        # add-one smoothed document-frequency rates
        ph = (high_counts[w] + 1) / (n_high_docs + 2)
        pl = (low_counts[w] + 1) / (n_low_docs + 2)
        enrichment = ph / pl
        # require some presence in high responses
        if high_counts[w] >= max(2, 0.1 * n_high_docs):
            scored.append((w, enrichment, high_counts[w], low_counts[w]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.PRIMARY_EVAL_MODELS)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()
    df = load_records(args.models)
    if df.empty:
        print("[differential_words] no records found")
        return
    rows = []
    for m in args.models:
        words = differential_words(df, m, top_n=args.top_n)
        rows.append(dict(model=m, words=", ".join(w for w, *_ in words)))
        print(f"\n{m}:\n  " + ", ".join(w for w, *_ in words))
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "table3_differential_words.csv",
                              index=False)


if __name__ == "__main__":
    main()
