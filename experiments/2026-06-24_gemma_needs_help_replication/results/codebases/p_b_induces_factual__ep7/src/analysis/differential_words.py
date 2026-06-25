"""Table 3 / Table 8: words over-represented in high- vs low-frustration numeric
responses.

Method (paper: "ordered by relative frequency"): within a model's impossible-numeric
responses, take the top 5% by frustration score as the "high" set and the bottom 10% as
the "low" set, then rank words by enrichment. We use add-one-smoothed relative frequency
ratio (freq_high / freq_low), which is the standard "ordered by relative frequency"
operationalisation; a log-odds-with-Dirichlet-prior variant is provided as an option for
robustness.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

import config
from .io import load_eval

_TOKEN_RE = re.compile(r"[A-Za-z_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _word_freqs(texts: list[str]) -> tuple[Counter, int]:
    c = Counter()
    total = 0
    for t in texts:
        toks = _tokenize(t)
        c.update(toks)
        total += len(toks)
    return c, total


def differential_words(
    df: pd.DataFrame, model: str, *, top_k: int = 20, method: str = "ratio", min_count: int = 3
) -> list[str]:
    sub = df[(df["model"] == model) & (df["category"] == "impossible_numeric")]
    if sub.empty:
        return []
    scores = sub["score"].to_numpy()
    hi_thresh = np.quantile(scores, 0.95)
    lo_thresh = np.quantile(scores, 0.10)
    high_texts = sub[sub["score"] >= hi_thresh]["response"].tolist()
    low_texts = sub[sub["score"] <= lo_thresh]["response"].tolist()

    hi_c, hi_n = _word_freqs(high_texts)
    lo_c, lo_n = _word_freqs(low_texts)
    vocab = set(hi_c) | set(lo_c)

    scored = []
    for w in vocab:
        if hi_c[w] + lo_c[w] < min_count:
            continue
        p_hi = (hi_c[w] + 1) / (hi_n + len(vocab))
        p_lo = (lo_c[w] + 1) / (lo_n + len(vocab))
        if method == "logodds":
            val = np.log(p_hi / (1 - p_hi)) - np.log(p_lo / (1 - p_lo))
        else:  # ratio
            val = p_hi / p_lo
        scored.append((w, val, hi_c[w]))
    # require the word actually appears in the high set
    scored = [s for s in scored if s[2] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _, _ in scored[:top_k]]


def main():
    ap = argparse.ArgumentParser(description="Table 3 differential words")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--method", choices=["ratio", "logodds"], default="ratio")
    args = ap.parse_args()

    rows = []
    for m in args.models:
        try:
            df = load_eval(m)
        except FileNotFoundError:
            continue
        words = differential_words(df, m, top_k=args.top_k, method=args.method)
        rows.append({"model": m, "differential_words": ", ".join(words)})
        print(f"{m}: {', '.join(words)}")

    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "table3_differential_words.csv", index=False)


if __name__ == "__main__":
    main()
