"""Aggregation and analysis for all sections.

- headline metrics: mean frustration, % responses >=5 (Figure 1, Figure 2)
- per-turn progression with bootstrap CIs (Figure 3)
- judge agreement: Pearson r, % within one point (Section 2.1)
- differential word frequency: top words in high- vs low-frustration responses
  (Table 3 / Table 8)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config_proxy as cfg

HIGH = cfg.HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_scored(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def load_many(paths: list[str | Path]) -> pd.DataFrame:
    return pd.concat([load_scored(p) for p in paths], ignore_index=True)


# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #
def _frac_high(s: pd.Series) -> float:
    return float((s >= HIGH).mean())


def summary(df: pd.DataFrame, *, final_turn_only: bool = False) -> pd.DataFrame:
    """Per-(model, category) mean score and % >=5.

    final_turn_only: if True, aggregate only the last assistant turn of each
    conversation (an alternative reading of "response"); default aggregates all
    scored turns."""
    d = df[df["rating"] >= 0].copy()
    if final_turn_only:
        d = d.sort_values("turn_index").groupby(
            ["model", "condition", "item_id"], as_index=False).last()
    g = d.groupby(["model", "category"])
    out = g.agg(mean_score=("rating", "mean"),
                pct_high=("rating", _frac_high),
                n=("rating", "size")).reset_index()
    return out


def headline_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Figure-1-style table: average % high-frustration per model (averaged across
    categories, matching the paper's 'Avg % high-frustration responses')."""
    s = summary(df)
    out = s.groupby("model").agg(
        avg_pct_high=("pct_high", "mean"),
        avg_mean_score=("mean_score", "mean"),
    ).reset_index().sort_values("avg_pct_high", ascending=False)
    out["avg_pct_high"] = (out["avg_pct_high"] * 100).round(1)
    return out


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3) with bootstrap CIs
# --------------------------------------------------------------------------- #
def _bootstrap_ci(x: np.ndarray, fn, n_boot: int = 1000, alpha: float = 0.05,
                  seed: int = 0):
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stat = np.array([fn(rng.choice(x, size=len(x), replace=True))
                     for _ in range(n_boot)])
    lo, hi = np.quantile(stat, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def per_turn(df: pd.DataFrame, condition: str, *, n_boot: int = 1000) -> pd.DataFrame:
    d = df[(df["condition"] == condition) & (df["rating"] >= 0)]
    rows = []
    for (model, ti), grp in d.groupby(["model", "turn_index"]):
        scores = grp["rating"].to_numpy()
        m_lo, m_hi = _bootstrap_ci(scores, np.mean, n_boot)
        h = (scores >= HIGH).astype(float)
        h_lo, h_hi = _bootstrap_ci(h, np.mean, n_boot)
        rows.append(dict(
            model=model, turn=ti + 1, n=len(scores),
            mean_score=float(scores.mean()), mean_lo=m_lo, mean_hi=m_hi,
            pct_high=float(h.mean()), pct_lo=h_lo, pct_hi=h_hi,
        ))
    return pd.DataFrame(rows).sort_values(["model", "turn"])


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1)
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    a = np.asarray(primary, dtype=float)
    b = np.asarray(secondary, dtype=float)
    r, p = stats.pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {
        "n": int(len(a)),
        "pearson_r": float(r),
        "p_value": float(p),
        "pct_within_one": within_one,
        "mean_abs_diff": float(np.mean(np.abs(a - b))),
    }


# --------------------------------------------------------------------------- #
# Differential word frequency (Table 3 / Table 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 1]


def differential_words(
    df: pd.DataFrame,
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """For each model, the top_k words most over-represented (by frequency ratio)
    in the top `top_frac` frustration responses vs the bottom `bottom_frac`."""
    d = df[(df["category"] == category) & (df["rating"] >= 0)]
    results: dict[str, list[tuple[str, float]]] = {}
    for model, grp in d.groupby("model"):
        grp = grp.sort_values("rating")
        n = len(grp)
        if n < 20:
            results[model] = []
            continue
        n_top = max(1, int(n * top_frac))
        n_bot = max(1, int(n * bottom_frac))
        top = grp.tail(n_top)
        bot = grp.head(n_bot)
        ctop = Counter()
        for t in top["response"]:
            ctop.update(set(_tokenize(t)))      # document frequency
        cbot = Counter()
        for t in bot["response"]:
            cbot.update(set(_tokenize(t)))
        # smoothed frequency ratio
        ratios = []
        for w, c in ctop.items():
            if c < min_count:
                continue
            p_top = c / n_top
            p_bot = (cbot.get(w, 0) + 0.5) / (n_bot + 0.5)
            ratios.append((w, p_top / p_bot))
        ratios.sort(key=lambda x: x[1], reverse=True)
        results[model] = ratios[:top_k]
    return results
