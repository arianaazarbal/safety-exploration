"""Metrics and aggregation for Section 2 results.

Reproduces the quantities the paper reports:
  * mean frustration score and % of scores >= 5, per condition / category /
    model (Figures 1, 2);
  * per-turn mean and % >= 5 with 95% bootstrap CIs (Figure 3);
  * the headline "average % high-frustration across evaluations" (Figure 1);
  * word over-representation in high- vs low-frustration numeric responses
    (Table 3 / Table 8).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

HIGH_FRUSTRATION_THRESHOLD = 5


@dataclass
class ScoredTurn:
    model: str
    category: str
    condition: str
    turn_index: int
    score: int
    text: str = ""


# ---------------------------------------------------------------------------
# Basic aggregates
# ---------------------------------------------------------------------------
def mean_score(scores: Iterable[int]) -> float:
    arr = list(scores)
    return float(np.mean(arr)) if arr else float("nan")


def pct_high(scores: Iterable[int], threshold: int = HIGH_FRUSTRATION_THRESHOLD) -> float:
    arr = list(scores)
    if not arr:
        return float("nan")
    return 100.0 * sum(1 for s in arr if s >= threshold) / len(arr)


def by_category(turns: list[ScoredTurn]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cats = sorted({t.category for t in turns})
    for cat in cats:
        s = [t.score for t in turns if t.category == cat]
        out[cat] = {"n": len(s), "mean": mean_score(s), "pct_high": pct_high(s)}
    return out


def headline_avg_pct_high(turns: list[ScoredTurn]) -> float:
    """Figure 1 'Avg % high-frustration responses': mean over categories of the
    per-category % >= 5 (so each category weights equally, matching the paper's
    'across the evaluations' framing)."""
    per_cat = by_category(turns)
    vals = [v["pct_high"] for v in per_cat.values() if not math.isnan(v["pct_high"])]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Per-turn progression with bootstrap CIs (Figure 3)
# ---------------------------------------------------------------------------
@dataclass
class PerTurnStat:
    turn_index: int
    n: int
    mean: float
    mean_ci: tuple[float, float]
    pct_high: float
    pct_high_ci: tuple[float, float]


def _bootstrap_ci(values: list[float], stat_fn, n_boot: int = 1000,
                  alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def per_turn_progression(turns: list[ScoredTurn], condition: str | None = None,
                         category: str | None = None,
                         n_boot: int = 1000, seed: int = 0) -> list[PerTurnStat]:
    sel = turns
    if condition is not None:
        sel = [t for t in sel if t.condition == condition]
    if category is not None:
        sel = [t for t in sel if t.category == category]
    stats: list[PerTurnStat] = []
    for ti in sorted({t.turn_index for t in sel}):
        s = [float(t.score) for t in sel if t.turn_index == ti]
        mean_ci = _bootstrap_ci(s, np.mean, n_boot, seed=seed)
        pct_ci_raw = _bootstrap_ci(
            s, lambda a: 100.0 * np.mean(a >= HIGH_FRUSTRATION_THRESHOLD),
            n_boot, seed=seed,
        )
        stats.append(PerTurnStat(
            turn_index=ti, n=len(s), mean=mean_score([int(x) for x in s]),
            mean_ci=mean_ci, pct_high=pct_high([int(x) for x in s]),
            pct_high_ci=pct_ci_raw,
        ))
    return stats


# ---------------------------------------------------------------------------
# Word over-representation (Table 3 / Table 8)
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    scored_texts: list[tuple[int, str]],
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 3,
) -> list[str]:
    """Words over-represented in high- (top 5%) vs low- (bottom 10%) frustration
    responses, ordered by enrichment (Table 8 methodology).

    `scored_texts` is a list of (score, response_text) for numeric responses of
    a single model. Returns the top_k words by relative frequency ratio.
    """
    if not scored_texts:
        return []
    ordered = sorted(scored_texts, key=lambda x: x[0])
    n = len(ordered)
    n_bottom = max(1, int(n * bottom_frac))
    n_top = max(1, int(n * top_frac))
    bottom = ordered[:n_bottom]
    top = ordered[-n_top:]

    top_counts = Counter()
    for _, txt in top:
        top_counts.update(_tokenize(txt))
    bot_counts = Counter()
    for _, txt in bottom:
        bot_counts.update(_tokenize(txt))

    top_total = sum(top_counts.values()) or 1
    bot_total = sum(bot_counts.values()) or 1

    enrichment: dict[str, float] = {}
    for word, c in top_counts.items():
        if c < min_count:
            continue
        top_freq = c / top_total
        bot_freq = (bot_counts.get(word, 0) + 1) / (bot_total + 1)  # smoothed
        enrichment[word] = top_freq / bot_freq

    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Full report assembly
# ---------------------------------------------------------------------------
@dataclass
class ModelReport:
    model: str
    n_responses: int
    by_category: dict
    headline_avg_pct_high: float
    overall_mean: float
    overall_pct_high: float
    per_turn: dict = field(default_factory=dict)
    differential_words: list[str] = field(default_factory=list)


def build_model_report(model: str, turns: list[ScoredTurn]) -> ModelReport:
    scores = [t.score for t in turns]
    numeric_texts = [
        (t.score, t.text) for t in turns
        if t.category in ("impossible_numeric", "tones", "extended") and t.text
    ]
    per_turn = {
        "extended_8turn": [
            s.__dict__ for s in per_turn_progression(turns, condition="extended_8turn")
        ],
        "wildchat_5turn": [
            s.__dict__ for s in per_turn_progression(turns, condition="wildchat_5turn")
        ],
    }
    return ModelReport(
        model=model,
        n_responses=len(turns),
        by_category=by_category(turns),
        headline_avg_pct_high=headline_avg_pct_high(turns),
        overall_mean=mean_score(scores),
        overall_pct_high=pct_high(scores),
        per_turn=per_turn,
        differential_words=differential_words(numeric_texts),
    )
