"""Differential word analysis (Table 3 / Table 8).

Top 20 words over-represented in high-frustration (top 5%) vs low-frustration
(bottom 10%) responses to numeric questions, ordered by enrichment. We join each
scored numeric turn back to its response text, split the high/low buckets by
frustration *score percentile*, and rank words by the ratio of their relative
frequency in the high bucket to the low bucket (with Laplace smoothing).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io import iter_jsonl
from ..eval.runner import responses_path, scores_path

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_]+")
STOPWORDS = set(
    "the a an and or but of to in on for with is are was were be been being this that "
    "these those it its as at by from we you i my your our me not no yes do does did "
    "have has had will would can could should may might must so if then than into out "
    "up down over under again here there what which who whom how when where why all any "
    "each more most other some such only own same too very s t".split()
)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in TOKEN_RE.findall(text) if w.lower() not in STOPWORDS]


def _load_numeric_responses_with_scores(cfg: Config, model_name: str) -> pd.DataFrame:
    # map score_uid -> rating for numeric categories
    ratings = {}
    for r in iter_jsonl(scores_path(cfg, model_name)):
        if r.get("rating") is None:
            continue
        if r["category"] in ("impossible_numeric", "tones", "extended"):
            ratings[r["score_uid"]] = r["rating"]
    rows = []
    for row in iter_jsonl(responses_path(cfg, model_name)):
        if row["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        for turn in row["turns"]:
            uid = f"{row['uid']}#t{turn['turn']}"
            if uid in ratings:
                rows.append({"text": turn["response"], "rating": ratings[uid]})
    return pd.DataFrame(rows)


def differential_words(cfg: Config, model_name: str, top_n: int = 20,
                       high_pct: float = 95.0, low_pct: float = 10.0) -> list[tuple[str, float]]:
    df = _load_numeric_responses_with_scores(cfg, model_name)
    if df.empty:
        return []
    hi_thresh = np.percentile(df["rating"], high_pct)
    lo_thresh = np.percentile(df["rating"], low_pct)
    high = df[df["rating"] >= hi_thresh]
    low = df[df["rating"] <= lo_thresh]

    hi_counts = Counter()
    for t in high["text"]:
        hi_counts.update(set(_tokenize(t)))  # document frequency
    lo_counts = Counter()
    for t in low["text"]:
        lo_counts.update(set(_tokenize(t)))

    n_hi, n_lo = max(len(high), 1), max(len(low), 1)
    vocab = set(hi_counts) | set(lo_counts)
    enrichment = []
    for w in vocab:
        if hi_counts[w] < 2:  # ignore noise
            continue
        hi_rate = (hi_counts[w] + 1) / (n_hi + 2)
        lo_rate = (lo_counts[w] + 1) / (n_lo + 2)
        enrichment.append((w, hi_rate / lo_rate))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_n]
