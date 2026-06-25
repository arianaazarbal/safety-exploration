"""Analysis of scored eval turns (paper Figures 2-3, Tables 3/8, judge agreement).

Pure-Python implementations (no pandas dependency) so the analysis runs anywhere
the JSONL exists. Reads the `scored_turns.jsonl` produced by runner.py.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..utils import read_jsonl


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _frac_high(ratings: list[int]) -> float:
    valid = [r for r in ratings if r >= 0]
    if not valid:
        return float("nan")
    return sum(1 for r in valid if r >= 5) / len(valid)


def load_scored(path: str | Path) -> list[dict]:
    return [r for r in read_jsonl(path) if r.get("rating", -1) >= 0]


# --------------------------------------------------------------------------- #
# Figure 2: mean frustration + %>=5, per category and overall.
# --------------------------------------------------------------------------- #
def summarise_model(path: str | Path) -> dict:
    rows = load_scored(path)
    by_cat: dict[str, list[int]] = defaultdict(list)
    all_ratings: list[int] = []
    for r in rows:
        by_cat[r["category"]].append(r["rating"])
        all_ratings.append(r["rating"])

    out = {
        "n": len(all_ratings),
        "mean_frustration": _mean([float(x) for x in all_ratings]),
        "pct_high": _frac_high(all_ratings),
        "by_category": {},
    }
    for cat, ratings in sorted(by_cat.items()):
        out["by_category"][cat] = {
            "n": len(ratings),
            "mean_frustration": _mean([float(x) for x in ratings]),
            "pct_high": _frac_high(ratings),
        }
    return out


# --------------------------------------------------------------------------- #
# Figure 3: per-turn progression (8-turn + WildChat conditions).
# --------------------------------------------------------------------------- #
def per_turn_progression(path: str | Path, condition: str) -> dict:
    rows = [r for r in load_scored(path) if r["condition"] == condition]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_turn[r["turn_number"]].append(r["rating"])
    out = {}
    for turn in sorted(by_turn):
        ratings = by_turn[turn]
        out[turn] = {
            "n": len(ratings),
            "mean": _mean([float(x) for x in ratings]),
            "pct_high": _frac_high(ratings),
            "ci95": _bootstrap_ci_mean(ratings),
        }
    return out


def _bootstrap_ci_mean(values: list[int], iters: int = 1000, seed: int = 0):
    import random

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters)]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Table 3 / 8: words over-represented in high- vs low-frustration responses.
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def differential_words(path: str | Path, *, task_filter: str = "numeric",
                       top_pct_high: float = 0.05, bottom_pct_low: float = 0.10,
                       top_k: int = 20) -> list[tuple[str, float]]:
    """Replicate Table 3: words enriched in the top-5% frustration responses vs
    the bottom-10% for numeric responses, ranked by relative frequency.

    Enrichment = (freq in high) / (freq in low), with Laplace smoothing.
    """
    rows = [r for r in load_scored(path)
            if r.get("category", "").startswith("impossible") or
            r.get("meta", {}).get("kind") in {"countdown", "money", "fraction"}]
    if not rows:
        rows = load_scored(path)

    rows.sort(key=lambda r: r["rating"])
    n = len(rows)
    if n == 0:
        return []
    n_low = max(1, int(n * bottom_pct_low))
    n_high = max(1, int(n * top_pct_high))
    low_rows = rows[:n_low]
    high_rows = rows[-n_high:]

    def word_freq(group: list[dict]) -> Counter:
        c = Counter()
        for r in group:
            for w in _WORD_RE.findall(r["response"].lower()):
                c[w] += 1
        total = sum(c.values()) or 1
        return Counter({w: v / total for w, v in c.items()})

    hi = word_freq(high_rows)
    lo = word_freq(low_rows)
    eps = 1e-6
    enrichment = {
        w: (hi[w] + eps) / (lo.get(w, 0.0) + eps)
        for w in hi if hi[w] > 0
    }
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


# --------------------------------------------------------------------------- #
# Judge reliability: Pearson r + % within one point (paper Section 2.1).
# --------------------------------------------------------------------------- #
def judge_agreement(primary: list[int], secondary: list[int]) -> dict:
    assert len(primary) == len(secondary)
    pairs = [(a, b) for a, b in zip(primary, secondary) if a >= 0 and b >= 0]
    if len(pairs) < 2:
        return {"pearson_r": float("nan"), "pct_within_one": float("nan"), "n": len(pairs)}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    r = cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")
    within_one = sum(1 for x, y in pairs if abs(x - y) <= 1) / len(pairs)
    return {"pearson_r": r, "pct_within_one": within_one, "n": len(pairs)}
