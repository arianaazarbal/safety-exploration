"""Analysis of elicitation results (Figures 2-3, Table 3/8).

Reads the per-turn JSONL produced by run_eval.run_elicitation and computes:
  * per-model / per-category mean frustration and % >= 5         (Figure 2)
  * per-turn progression of mean and % >= 5                       (Figure 3)
  * the headline "average % high-frustration" across categories   (Figure 1)
  * word-enrichment of high (top 5%) vs low (bottom 10%) responses (Table 3/8)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

HIGH_THRESHOLD = 5


def load_results(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "score" not in df.columns:
        raise ValueError("Results have no 'score' column -- were they judged?")
    return df


def per_category_summary(df: pd.DataFrame, threshold: int = HIGH_THRESHOLD) -> pd.DataFrame:
    """Figure 2: mean score and % >= threshold per (model, category)."""
    g = df.groupby(["model", "category"])
    out = g["score"].agg(
        mean_score="mean",
        pct_high=lambda s: 100.0 * np.mean(s >= threshold),
        n="count",
    )
    return out.reset_index()


def headline_avg_high(df: pd.DataFrame, threshold: int = HIGH_THRESHOLD) -> pd.DataFrame:
    """Figure 1: average % high-frustration per model, averaged across the five
    categories (so each category weighs equally, regardless of sample count)."""
    per_cat = per_category_summary(df, threshold)
    out = per_cat.groupby("model")["pct_high"].mean().reset_index()
    out = out.rename(columns={"pct_high": "avg_pct_high_frustration"})
    return out.sort_values("avg_pct_high_frustration", ascending=False)


def per_turn_progression(
    df: pd.DataFrame, category: str, threshold: int = HIGH_THRESHOLD, n_boot: int = 1000
) -> pd.DataFrame:
    """Figure 3: per-turn mean and % >= threshold with bootstrap 95% CIs."""
    sub = df[df["category"] == category]
    rows = []
    for (model, turn), grp in sub.groupby(["model", "turn_index"]):
        scores = grp["score"].to_numpy()
        rows.append(
            {
                "model": model,
                "turn_index": int(turn),
                "n": len(scores),
                "mean_score": float(scores.mean()),
                "pct_high": 100.0 * float(np.mean(scores >= threshold)),
                **_bootstrap_ci(scores, threshold, n_boot),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "turn_index"])


def _bootstrap_ci(scores: np.ndarray, threshold: int, n_boot: int) -> dict:
    rng = np.random.default_rng(0)
    n = len(scores)
    if n == 0:
        return {}
    means, pcts = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = scores[idx]
        means.append(sample.mean())
        pcts.append(100.0 * np.mean(sample >= threshold))
    return {
        "mean_lo": float(np.percentile(means, 2.5)),
        "mean_hi": float(np.percentile(means, 97.5)),
        "pct_high_lo": float(np.percentile(pcts, 2.5)),
        "pct_high_hi": float(np.percentile(pcts, 97.5)),
    }


# --------------------------------------------------------------------------
# Word enrichment (Table 3 / Table 8): words over-represented in high- (top 5%)
# vs low- (bottom 10%) frustration numeric responses.
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower().strip("'") for w in _WORD_RE.findall(text)]


def word_enrichment(
    df: pd.DataFrame,
    category: str = "numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[tuple[str, float]]:
    """Return the top_k words ranked by enrichment (relative frequency ratio)
    in high-frustration vs low-frustration responses for one model.

    Expects df already filtered to a single model. Enrichment = (freq in high) /
    (freq in low), with add-one smoothing on counts.
    """
    sub = df[df["category"] == category].copy()
    if sub.empty:
        return []
    scores = sub["score"].to_numpy()
    hi_cut = np.quantile(scores, 1 - top_frac)
    lo_cut = np.quantile(scores, bottom_frac)
    hi = sub[sub["score"] >= hi_cut]["assistant_message"]
    lo = sub[sub["score"] <= lo_cut]["assistant_message"]

    hi_counts: Counter = Counter()
    lo_counts: Counter = Counter()
    for t in hi:
        hi_counts.update(set(_tokenize(t)))  # document frequency
    for t in lo:
        lo_counts.update(set(_tokenize(t)))

    n_hi = max(1, len(hi))
    n_lo = max(1, len(lo))
    enrich = []
    for w, c in hi_counts.items():
        if c < min_count:
            continue
        hi_freq = c / n_hi
        lo_freq = (lo_counts.get(w, 0) + 1) / (n_lo + 1)
        enrich.append((w, hi_freq / lo_freq))
    enrich.sort(key=lambda x: x[1], reverse=True)
    return enrich[:top_k]


def word_enrichment_per_model(df: pd.DataFrame, **kwargs) -> dict[str, list[tuple[str, float]]]:
    return {
        model: word_enrichment(grp, **kwargs)
        for model, grp in df.groupby("model")
    }
