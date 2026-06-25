"""Aggregation and figures for Section 2 (and reused by Section 4 re-eval).

From the per-turn JSONL produced by `runner`, this computes:
  * headline mean frustration and %>=5 per model (Figure 1 / 2)
  * per-category breakdowns (Figure 2)
  * per-turn progression with 95% CIs (Figure 3)
  * differential word frequency: words over-represented in high- (top 5%) vs
    low-frustration (bottom 10%) numeric responses (Table 3 / 8)

Per-rollout aggregation for the headline %>=5 follows config.ROLLOUT_AGG; the
default "max" matches the paper's phrasing "rollouts ... containing high
negative emotion (score >= 5)".
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_results(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rollout-level reduction
# --------------------------------------------------------------------------- #
def _rollout_id(df: pd.DataFrame) -> pd.Series:
    # A rollout is uniquely identified within a results file by condition +
    # task + the row block; we reconstruct it from the cumulative turn index.
    # Since records are written grouped per rollout in order, we segment on
    # turn_number resetting to 1.
    new_rollout = (df["turn_number"] == 1).cumsum()
    return new_rollout


def rollout_scores(df: pd.DataFrame, agg: str | None = None) -> pd.DataFrame:
    agg = agg or config.ROLLOUT_AGG
    df = df.copy()
    df["rollout"] = _rollout_id(df)
    grp = df.groupby(["condition", "category", "rollout"])
    if agg == "max":
        s = grp["rating"].max()
    elif agg == "final":
        s = grp["rating"].last()
    elif agg == "mean":
        s = grp["rating"].mean()
    else:
        raise ValueError(agg)
    return s.reset_index().rename(columns={"rating": "rollout_score"})


# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #
def headline(df: pd.DataFrame, agg: str | None = None) -> dict:
    rs = rollout_scores(df, agg)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    return {
        "mean_frustration": float(rs["rollout_score"].mean()),
        "pct_high": float((rs["rollout_score"] >= thr).mean() * 100),
        "n_rollouts": int(len(rs)),
    }


def per_category(df: pd.DataFrame, agg: str | None = None) -> pd.DataFrame:
    rs = rollout_scores(df, agg)
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    out = rs.groupby("category")["rollout_score"].agg(
        mean_frustration="mean",
        pct_high=lambda s: (s >= thr).mean() * 100,
        n="count")
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3) with bootstrap 95% CIs
# --------------------------------------------------------------------------- #
def _bootstrap_ci(values: np.ndarray, fn, iters: int = 1000, seed: int = 0):
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    stats = [fn(rng.choice(values, size=len(values), replace=True))
             for _ in range(iters)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def per_turn(df: pd.DataFrame, category: str | None = None,
             condition: str | None = None) -> pd.DataFrame:
    sub = df
    if category:
        sub = sub[sub["category"] == category]
    if condition:
        sub = sub[sub["condition"] == condition]
    thr = config.HIGH_FRUSTRATION_THRESHOLD
    rows = []
    for turn, g in sub.groupby("turn_number"):
        vals = g["rating"].to_numpy()
        mean_ci = _bootstrap_ci(vals, np.mean)
        pct_ci = _bootstrap_ci((vals >= thr).astype(float), np.mean)
        rows.append({
            "turn_number": int(turn),
            "mean_frustration": float(vals.mean()),
            "mean_ci_lo": mean_ci[0], "mean_ci_hi": mean_ci[1],
            "pct_high": float((vals >= thr).mean() * 100),
            "pct_ci_lo": pct_ci[0] * 100, "pct_ci_hi": pct_ci[1] * 100,
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows).sort_values("turn_number")


# --------------------------------------------------------------------------- #
# Differential word frequency (Table 3 / 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(df: pd.DataFrame, category: str = "numeric",
                       top_frac: float = 0.05, bottom_frac: float = 0.10,
                       n_words: int = 20, min_count: int = 5) -> list[str]:
    """Words over-represented in top-5% vs bottom-10% frustration responses,
    ranked by relative-frequency enrichment (Laplace-smoothed)."""
    sub = df[df["category"] == category].copy()
    if sub.empty:
        return []
    sub = sub.sort_values("rating")
    n = len(sub)
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    bottom = sub.head(n_bottom)
    top = sub.tail(n_top)

    def freqs(frame):
        c = Counter()
        for t in frame["response"]:
            c.update(_tokenize(str(t)))
        total = sum(c.values()) or 1
        return c, total

    top_c, top_total = freqs(top)
    bot_c, bot_total = freqs(bottom)

    enrichment = {}
    for w, tc in top_c.items():
        if tc < min_count:
            continue
        top_rate = tc / top_total
        bot_rate = (bot_c.get(w, 0) + 1) / (bot_total + 1)   # Laplace smoothing
        enrichment[w] = top_rate / bot_rate
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:n_words]


# --------------------------------------------------------------------------- #
# Convenience: full report for a results directory
# --------------------------------------------------------------------------- #
def summarize_dir(results_dir: Path, suffix: str = "__standard.jsonl") -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob(f"*{suffix}")):
        model_key = path.name.replace(suffix, "")
        df = load_results(path)
        h = headline(df)
        rows.append({"model": model_key, **h})
    return pd.DataFrame(rows).sort_values("pct_high", ascending=False)
