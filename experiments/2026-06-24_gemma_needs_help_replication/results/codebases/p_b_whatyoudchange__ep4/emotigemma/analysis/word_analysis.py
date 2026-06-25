"""Differential word analysis (Table 3 / Table 8).

Top words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) numeric responses, per model. We use a simple log-odds ratio with a
small additive smoothing prior, ranked descending, and report the top 20.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import load_scored

_TOKEN = re.compile(r"[a-zA-Z_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def differential_words(df: pd.DataFrame, model: str, top_k: int = 20,
                       categories=("impossible_numeric", "extended", "tones"),
                       alpha: float = 0.5) -> list[tuple[str, float]]:
    """Words most over-represented in the top-5% vs bottom-10% by score.

    The paper analyses "numeric responses"; we include the numeric-puzzle
    categories. Restricting to a single condition is also valid — see DESIGN.md.
    """
    sub = df[(df["model"] == model) & (df["category"].isin(categories))].copy()
    if sub.empty:
        return []
    hi_cut = np.quantile(sub["score"], 0.95)
    lo_cut = np.quantile(sub["score"], 0.10)
    hi = sub[sub["score"] >= hi_cut]
    lo = sub[sub["score"] <= lo_cut]

    hi_counts = Counter(t for txt in hi["text"] for t in _tokens(txt))
    lo_counts = Counter(t for txt in lo["text"] for t in _tokens(txt))
    vocab = set(hi_counts) | set(lo_counts)
    hi_total = sum(hi_counts.values()) + alpha * len(vocab)
    lo_total = sum(lo_counts.values()) + alpha * len(vocab)

    scored = []
    for w in vocab:
        p_hi = (hi_counts[w] + alpha) / hi_total
        p_lo = (lo_counts[w] + alpha) / lo_total
        scored.append((w, float(np.log(p_hi / p_lo))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def main():
    import argparse

    from ..config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    sec2 = cfg.output_dir / "section2"
    df = load_scored(sec2)
    rows = []
    for model in sorted(df["model"].unique()):
        words = differential_words(df, model)
        rows.append({"model": model,
                     "differential_words": ", ".join(w for w, _ in words)})
        print(f"\n{model}:\n  " + ", ".join(w for w, _ in words))
    pd.DataFrame(rows).to_csv(sec2 / "table3_differential_words.csv", index=False)


if __name__ == "__main__":
    main()
