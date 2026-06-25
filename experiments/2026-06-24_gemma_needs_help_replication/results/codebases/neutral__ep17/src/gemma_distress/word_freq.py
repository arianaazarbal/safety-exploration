"""Table 3 / Table 8: words over-represented in high- vs low-frustration
responses to numeric questions.

For each model we take its impossible-numeric responses, split into the top 5%
(high) and bottom 10%(low) by frustration score, and rank words by enrichment
(relative frequency in high vs low). We report the top 20.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .analysis import load_all_scores

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_STOP = set("the a an and or of to in is it i you me my we is are was be that this "
            "for on with as at by from your s t re ve ll".split())


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text) if w.lower() not in _STOP and len(w) > 1]


def differential_words(df: pd.DataFrame, model: str, top_k: int = 20,
                       high_q: float = 0.95, low_q: float = 0.10) -> list[str]:
    sub = df[(df["model"] == model) & (df["category"] == "impossible_numeric")]
    if sub.empty:
        return []
    ratings = sub["rating"].to_numpy()
    hi_thresh = pd.Series(ratings).quantile(high_q)
    lo_thresh = pd.Series(ratings).quantile(low_q)
    high = sub[sub["rating"] >= hi_thresh]
    low = sub[sub["rating"] <= lo_thresh]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for t in high["response"]:
        hi_counts.update(_tokens(t))
    for t in low["response"]:
        lo_counts.update(_tokens(t))
    hi_total = sum(hi_counts.values()) or 1
    lo_total = sum(lo_counts.values()) or 1

    enrich = {}
    for w, c in hi_counts.items():
        if c < 3:
            continue
        hi_freq = c / hi_total
        lo_freq = (lo_counts.get(w, 0) + 1) / (lo_total + 1)
        enrich[w] = hi_freq / lo_freq
    return [w for w, _ in sorted(enrich.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


def build_table(scores_dir: Path, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = load_all_scores(scores_dir)
    rows = []
    for model in sorted(df["model"].unique()):
        words = differential_words(df, model)
        rows.append({"model": model, "differential_words": ", ".join(words)})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out_csv
