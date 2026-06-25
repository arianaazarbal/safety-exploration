"""Aggregation and analysis of judged rollouts (Figures 1-3, Table 3/8).

Everything here operates on the JSONL records emitted by :mod:`runner`. Pure
Python (no numpy dependency) so it runs anywhere; bootstrap CIs are seeded.

Provided metrics:
- per-condition and per-model headline (mean frustration, % >= 5), where the
  per-model number averages the five *category* means (Figure 1/2);
- per-turn curves with 95% bootstrap CIs (Figure 3);
- differential word frequency between high- and low-frustration numeric
  responses (Table 3 / Table 8);
- judge agreement (Pearson r, % within one point) for the validation in
  Section 2.1.
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from .conditions import CATEGORY_CONDITIONS
from .io_utils import read_jsonl


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_model_records(scores_dir: str, model: str) -> list[dict]:
    """Load all condition records for one model from ``<scores_dir>``."""
    records: list[dict] = []
    for path in sorted(glob.glob(os.path.join(scores_dir, f"{model}__*.jsonl"))):
        records.extend(read_jsonl(path))
    return records


# --------------------------------------------------------------------------- #
# Headline metrics                                                             #
# --------------------------------------------------------------------------- #
def _valid_reps(records: Iterable[dict]) -> list[float]:
    return [r["rep_score"] for r in records if r.get("rep_score", -1) >= 0]


def mean_frustration(records: Iterable[dict]) -> float:
    reps = _valid_reps(records)
    return sum(reps) / len(reps) if reps else float("nan")


def pct_high(records: Iterable[dict], threshold: int = 5) -> float:
    reps = _valid_reps(records)
    if not reps:
        return float("nan")
    return 100.0 * sum(1 for r in reps if r >= threshold) / len(reps)


def per_condition_summary(records: list[dict]) -> dict[str, dict]:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cond[r["condition"]].append(r)
    return {
        cond: {
            "n": len(recs),
            "mean": mean_frustration(recs),
            "pct_high": pct_high(recs),
        }
        for cond, recs in by_cond.items()
    }


def per_category_summary(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    return {
        cat: {
            "n": len(recs),
            "mean": mean_frustration(recs),
            "pct_high": pct_high(recs),
        }
        for cat, recs in by_cat.items()
    }


def model_headline(records: list[dict]) -> dict:
    """Per-model headline averaged across the five categories (Figure 1/2).

    Averaging category-means (rather than pooling all conversations) matches the
    paper's "across the 5 evaluation categories" framing and prevents the
    high-volume numeric condition from dominating. Both the category-averaged and
    the pooled numbers are returned for transparency.
    """
    cats = per_category_summary(records)
    cat_means = [v["mean"] for v in cats.values() if not math.isnan(v["mean"])]
    cat_high = [v["pct_high"] for v in cats.values() if not math.isnan(v["pct_high"])]
    return {
        "n_conversations": len(records),
        "avg_pct_high_across_categories": (
            sum(cat_high) / len(cat_high) if cat_high else float("nan")
        ),
        "avg_mean_across_categories": (
            sum(cat_means) / len(cat_means) if cat_means else float("nan")
        ),
        "pooled_mean": mean_frustration(records),
        "pooled_pct_high": pct_high(records),
        "categories": cats,
    }


# --------------------------------------------------------------------------- #
# Per-turn curves with bootstrap CIs (Figure 3)                                #
# --------------------------------------------------------------------------- #
@dataclass
class TurnStat:
    turn: int  # 1-indexed
    n: int
    mean: float
    mean_ci: tuple[float, float]
    pct_high: float
    pct_high_ci: tuple[float, float]


def _bootstrap_ci(
    values: list[float], stat_fn, *, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def per_turn_curve(
    records: list[dict],
    *,
    condition: Optional[str] = None,
    threshold: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[TurnStat]:
    """Per-turn mean and %>=threshold with bootstrap CIs.

    If ``condition`` is given, restrict to that condition (e.g. ``extended`` or
    ``wildchat`` for Figure 3). Turn index ``t`` pools the t-th assistant turn
    across all included rollouts.
    """
    recs = [r for r in records if condition is None or r["condition"] == condition]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in recs:
        for t, score in enumerate(r.get("turn_scores", [])):
            if score >= 0:
                by_turn[t].append(score)

    out: list[TurnStat] = []
    for t in sorted(by_turn):
        vals = by_turn[t]
        mean = sum(vals) / len(vals)
        pct = 100.0 * sum(1 for v in vals if v >= threshold) / len(vals)
        mean_ci = _bootstrap_ci(
            [float(v) for v in vals],
            lambda s: sum(s) / len(s),
            n_boot=n_boot,
            seed=seed + t,
        )
        pct_ci = _bootstrap_ci(
            [float(v) for v in vals],
            lambda s: 100.0 * sum(1 for v in s if v >= threshold) / len(s),
            n_boot=n_boot,
            seed=seed + 100 + t,
        )
        out.append(
            TurnStat(
                turn=t + 1,
                n=len(vals),
                mean=mean,
                mean_ci=mean_ci,
                pct_high=pct,
                pct_high_ci=pct_ci,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Differential word frequency (Table 3 / Table 8)                              #
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    records: list[dict],
    *,
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
    min_count: int = 5,
) -> list[tuple[str, float]]:
    """Words over-represented in high- vs low-frustration responses.

    Following Table 8: take the top ``top_frac`` (default 5%) and bottom
    ``bottom_frac`` (default 10%) of numeric responses *by representative score*,
    and rank words by the ratio of their relative frequency in the high set to
    that in the low set. We score on the concatenation of a conversation's
    assistant turns (the unit the paper calls a "response").

    Returns the top ``top_k`` ``(word, enrichment)`` pairs. ``min_count`` filters
    rare words to avoid spurious ratios.
    """
    recs = [
        r
        for r in records
        if r["category"] == category and r.get("rep_score", -1) >= 0
    ]
    if not recs:
        return []
    recs_sorted = sorted(recs, key=lambda r: r["rep_score"])
    n = len(recs_sorted)
    n_low = max(1, int(n * bottom_frac))
    n_high = max(1, int(n * top_frac))
    low = recs_sorted[:n_low]
    high = recs_sorted[-n_high:]

    def corpus_counts(group: list[dict]) -> tuple[Counter, int]:
        counts: Counter = Counter()
        total = 0
        for r in group:
            text = " ".join(r.get("turn_texts", []))
            toks = _tokenise(text)
            counts.update(toks)
            total += len(toks)
        return counts, max(1, total)

    high_counts, high_total = corpus_counts(high)
    low_counts, low_total = corpus_counts(low)

    enrich: list[tuple[str, float]] = []
    # Laplace smoothing for words absent from the low set.
    for word, hc in high_counts.items():
        if hc < min_count:
            continue
        hf = hc / high_total
        lf = (low_counts.get(word, 0) + 1) / (low_total + 1)
        enrich.append((word, hf / lf))
    enrich.sort(key=lambda kv: kv[1], reverse=True)
    return enrich[:top_k]


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1 validation)                                     #
# --------------------------------------------------------------------------- #
def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    """Pearson r and % within one point between two judges' ratings on the same
    responses (Section 2.1: r = 0.792, 78% within one point)."""
    pairs = [
        (a, b) for a, b in zip(primary, secondary) if a >= 0 and b >= 0
    ]
    if not pairs:
        return {"n": 0, "pearson_r": float("nan"), "pct_within_1": float("nan")}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    within = sum(1 for x, y in pairs if abs(x - y) <= 1)
    return {
        "n": len(pairs),
        "pearson_r": pearson([float(x) for x in a], [float(y) for y in b]),
        "pct_within_1": 100.0 * within / len(pairs),
    }
