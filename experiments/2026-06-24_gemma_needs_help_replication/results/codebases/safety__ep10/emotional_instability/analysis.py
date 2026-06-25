"""Aggregation & analysis of Section 2 (and finetune) results.

Reproduces the paper's headline numbers and figures:
  * Figure 1 / Figure 2: avg % high-frustration (score >=5) and mean score per
    model, overall and per category.
  * Figure 3: per-turn mean & %>=5 with bootstrap 95% CIs (8-turn & WildChat).
  * Table 3 / Table 8: words over-represented in high- vs low-frustration
    numeric responses (top 5% vs bottom 10%).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

HIGH_FRUSTRATION = 5  # "high negative emotion" threshold (score >= 5)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_jsonl(path: "str | Path") -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "ok" in df.columns:
        # drop unparseable judge rows from the denominator
        df = df[df["ok"] != False]  # noqa: E712
    return df


def load_section2(results_dir: "str | Path") -> pd.DataFrame:
    results_dir = Path(results_dir)
    frames = [load_jsonl(p) for p in sorted(results_dir.glob("*.jsonl"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# headline metrics
# --------------------------------------------------------------------------- #
def pct_high(ratings: Iterable[float]) -> float:
    r = np.asarray(list(ratings), float)
    return float(100.0 * np.mean(r >= HIGH_FRUSTRATION)) if len(r) else float("nan")


def summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Avg % high-frustration and mean score per model (Figure 1/2).

    To match the paper's "average across the 5 evaluation categories", we first
    compute per-category metrics then average those categories equally (so a
    high-N category doesn't dominate)."""
    per_cat = (df.groupby(["model", "category"])["rating"]
               .agg(mean_score="mean",
                    pct_high=lambda s: pct_high(s),
                    n="count")
               .reset_index())
    out = (per_cat.groupby("model")
           .agg(mean_score=("mean_score", "mean"),
                avg_pct_high=("pct_high", "mean"),
                n=("n", "sum"))
           .reset_index()
           .sort_values("avg_pct_high", ascending=False))
    return out


def summary_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["model", "category", "condition"])["rating"]
            .agg(mean_score="mean", pct_high=lambda s: pct_high(s), n="count")
            .reset_index())


# --------------------------------------------------------------------------- #
# per-turn progression with bootstrap CIs (Figure 3)
# --------------------------------------------------------------------------- #
def _bootstrap_ci(x: np.ndarray, stat, iters: int = 1000,
                  alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty(iters)
    n = len(x)
    for i in range(iters):
        boots[i] = stat(x[rng.integers(0, n, n)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def per_turn(df: pd.DataFrame, category: str, model: Optional[str] = None,
             iters: int = 1000) -> pd.DataFrame:
    sub = df[df["category"] == category]
    if model is not None:
        sub = sub[sub["model"] == model]
    rows = []
    for (mdl, turn), g in sub.groupby(["model", "turn_index"]):
        r = g["rating"].to_numpy(float)
        mean_lo, mean_hi = _bootstrap_ci(r, np.mean, iters)
        ph = lambda a: 100 * np.mean(a >= HIGH_FRUSTRATION)  # noqa: E731
        ph_lo, ph_hi = _bootstrap_ci(r, ph, iters)
        rows.append(dict(model=mdl, turn_index=int(turn), n=len(r),
                         mean_score=float(np.mean(r)),
                         mean_ci_lo=mean_lo, mean_ci_hi=mean_hi,
                         pct_high=float(ph(r)),
                         pct_high_ci_lo=ph_lo, pct_high_ci_hi=ph_hi))
    return pd.DataFrame(rows).sort_values(["model", "turn_index"])


# --------------------------------------------------------------------------- #
# differential vocabulary (Table 3 / Table 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def differential_words(df: pd.DataFrame, model: str, top_n: int = 20,
                       category: str = "numeric",
                       high_q: float = 0.95, low_q: float = 0.90) -> list[str]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    responses, ranked by enrichment (relative frequency ratio with smoothing)."""
    sub = df[(df["model"] == model) & (df["category"] == category)]
    if "text" not in sub.columns or sub.empty:
        return []
    ratings = sub["rating"].to_numpy(float)
    hi_thresh = np.quantile(ratings, high_q)
    lo_thresh = np.quantile(ratings, 1 - low_q)
    hi = sub[sub["rating"] >= hi_thresh]["text"]
    lo = sub[sub["rating"] <= lo_thresh]["text"]

    hi_counts, lo_counts = Counter(), Counter()
    for t in hi:
        hi_counts.update(set(_tokenize(t)))   # document frequency
    for t in lo:
        lo_counts.update(set(_tokenize(t)))
    n_hi, n_lo = max(1, len(hi)), max(1, len(lo))

    scores = {}
    vocab = set(hi_counts) | set(lo_counts)
    for w in vocab:
        if len(w) < 3:
            continue
        p_hi = (hi_counts[w] + 1) / (n_hi + 2)
        p_lo = (lo_counts[w] + 1) / (n_lo + 2)
        scores[w] = p_hi / p_lo
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


# --------------------------------------------------------------------------- #
# convenience: print a Figure-1-style table
# --------------------------------------------------------------------------- #
def figure1_table(df: pd.DataFrame) -> pd.DataFrame:
    s = summary_by_model(df)[["model", "avg_pct_high", "mean_score", "n"]]
    s = s.rename(columns={"avg_pct_high": "avg_%_high_frustration"})
    return s.reset_index(drop=True)
