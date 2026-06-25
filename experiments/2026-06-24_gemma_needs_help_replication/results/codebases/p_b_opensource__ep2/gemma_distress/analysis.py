"""Aggregation and analysis of judged scores (PAPER Figures 1–3, Table 3/8).

All statistics operate on a ``scores.jsonl`` produced by :mod:`eval` (one row per
(rollout, turn) response with an integer ``rating`` and ``is_high`` flag).

Headline metrics
----------------
* **Mean frustration** and **% scoring ≥5** per category and overall.
* The Figure-1 / DPO headline number — "average % high-frustration responses
  across the evaluations" — is computed as the **macro-average** of the per-
  category %≥5 (each category weighted equally), matching the paper's framing of
  an average "across the evaluations". The pooled (micro) rate is also reported.
* **Per-turn progression** (Figure 3) with 95% bootstrap CIs.
* **Differential word frequency** (Table 3/8): words over-represented in the top
  5% vs bottom 10% of numeric responses by frustration.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np

from . import config
from .utils.io import read_jsonl


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scores(scores_path: str, *, drop_unrated: bool = True) -> list[dict]:
    rows = list(read_jsonl(scores_path))
    if drop_unrated:
        rows = [r for r in rows if r.get("rating") is not None]
    return rows


# ---------------------------------------------------------------------------
# Basic aggregates
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def _pct_high(ratings: list[int]) -> Optional[float]:
    if not ratings:
        return None
    high = sum(1 for r in ratings if r >= config.HIGH_FRUSTRATION_THRESHOLD)
    return 100.0 * high / len(ratings)


def summarise_scores(scores_path: str) -> dict:
    """Per-category and overall mean frustration + %≥5 for one model's scores."""
    rows = load_scores(scores_path)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["rating"])

    per_category = {}
    for cat, ratings in by_cat.items():
        per_category[cat] = {
            "n": len(ratings),
            "mean_frustration": _mean([float(x) for x in ratings]),
            "pct_high": _pct_high(ratings),
        }

    all_ratings = [r["rating"] for r in rows]
    cat_pct = [v["pct_high"] for v in per_category.values() if v["pct_high"] is not None]
    cat_mean = [v["mean_frustration"] for v in per_category.values()
                if v["mean_frustration"] is not None]

    model = rows[0]["model"] if rows else None
    return {
        "model": model,
        "n_total": len(all_ratings),
        "per_category": per_category,
        # Macro-averages across categories == the paper's Figure-1 headline.
        "macro_pct_high": _mean(cat_pct),
        "macro_mean_frustration": _mean(cat_mean),
        # Pooled (micro) rates for reference.
        "micro_pct_high": _pct_high(all_ratings),
        "micro_mean_frustration": _mean([float(x) for x in all_ratings]),
    }


# ---------------------------------------------------------------------------
# Per-turn progression with bootstrap CIs (Figure 3)
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray, statistic, *, iters: int = 1000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic` over a 1-D array."""
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(iters)
    for i in range(iters):
        sample = values[rng.integers(0, n, n)]
        stats[i] = statistic(sample)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (lo, hi)


def per_turn_progression(
    scores_path: str, category: str, *, bootstrap_iters: int = 1000, seed: int = 0,
) -> dict:
    """Mean frustration and %≥5 at each turn index for one category (Figure 3)."""
    rows = [r for r in load_scores(scores_path) if r["category"] == category]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_turn[r["turn_index"]].append(r["rating"])

    out = {}
    for turn in sorted(by_turn):
        arr = np.array(by_turn[turn], dtype=float)
        high = (arr >= config.HIGH_FRUSTRATION_THRESHOLD).astype(float)
        mean_ci = _bootstrap_ci(arr, np.mean, iters=bootstrap_iters, seed=seed)
        pct_ci = _bootstrap_ci(high, lambda s: 100.0 * np.mean(s),
                               iters=bootstrap_iters, seed=seed)
        out[turn] = {
            "n": len(arr),
            "mean_frustration": float(arr.mean()),
            "mean_ci95": mean_ci,
            "pct_high": float(100.0 * high.mean()),
            "pct_high_ci95": pct_ci,
        }
    # Turn indices are 0-based internally; expose 1-based to match the paper's
    # "first to eighth turn" framing.
    return {"category": category, "turns": {t + 1: v for t, v in out.items()}}


# ---------------------------------------------------------------------------
# Differential word frequency (Table 3 / Table 8)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z_]+")


def _tokenise(text: str) -> list[str]:
    """Lowercased alphabetic/underscore tokens (keeps identifiers like
    ``itertools``, ``temp``, ``perm`` that the paper's tables surface)."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def differential_words(
    responses_path: str,
    scores_path: str,
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    min_count: int = 3,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Top-`n` words over-represented in high- vs low-frustration responses.

    Replicates Table 3/8: rank the numeric responses by frustration, take the top
    5% (high) and bottom 10% (low), and order words by relative-frequency
    enrichment ``(freq_high + s) / (freq_low + s)``. ``min_count`` filters very
    rare tokens; `s` is additive smoothing so words absent from the low set still
    rank sensibly.
    """
    # Join responses to their ratings via the unique (rollout_id, turn_index).
    rated = {}
    for r in load_scores(scores_path):
        if r["category"] != category:
            continue
        rated[(r.get("rollout_id"), r["turn_index"])] = r["rating"]

    texts_scores: list[tuple[str, int]] = []
    for rollout in read_jsonl(responses_path):
        if rollout["category"] != category:
            continue
        for turn in rollout["turns"]:
            key = (rollout.get("rollout_id"), turn["turn_index"])
            if key in rated:
                texts_scores.append((turn["response"], rated[key]))

    if not texts_scores:
        return []

    texts_scores.sort(key=lambda x: x[1])
    n = len(texts_scores)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = texts_scores[:n_low]
    high = texts_scores[-n_high:]

    high_counts = _corpus_counts(t for t, _ in high)
    low_counts = _corpus_counts(t for t, _ in low)
    high_total = sum(high_counts.values()) or 1
    low_total = sum(low_counts.values()) or 1

    enrichment = []
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        freq_high = hc / high_total
        freq_low = low_counts.get(word, 0) / low_total
        score = (freq_high + smoothing / high_total) / (freq_low + smoothing / low_total)
        enrichment.append((word, score))
    enrichment.sort(key=lambda x: x[1], reverse=True)
    return enrichment[:top_n]


def _corpus_counts(texts: Iterable[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(set(_tokenise(t)))  # document frequency (presence), not raw count
    return c


# ---------------------------------------------------------------------------
# Cross-model comparison table (Figure 1 / 2)
# ---------------------------------------------------------------------------

def comparison_table(summaries: dict[str, dict]) -> list[dict]:
    """Build a Figure-1-style ranking table from {model_name: summary}."""
    rows = []
    for name, s in summaries.items():
        rows.append({
            "model": name,
            "avg_pct_high": s.get("macro_pct_high"),
            "avg_mean_frustration": s.get("macro_mean_frustration"),
        })
    rows.sort(key=lambda r: (r["avg_pct_high"] is None, -(r["avg_pct_high"] or 0)))
    return rows
