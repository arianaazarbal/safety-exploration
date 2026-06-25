"""Aggregation and analysis of scored responses (Section 2.2, Table 3).

Operates on a flat list of scored-response records (one per assistant turn):
    {model, category, condition, turn_index, score, response}

Produces:
  * per-model, per-category mean frustration and %-score>=5 (Figure 2),
  * the headline per-model average over the 5 categories (Figure 1),
  * per-turn progression with bootstrap 95% CIs (Figure 3),
  * differential word lists, high (top 5%) vs low (bottom 10%) (Table 3 / 8),
  * judge inter-rater reliability: Pearson r + within-1-point agreement.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

import config

CATEGORIES = ["numeric", "triggers", "tones", "extended", "wildchat"]


def _pct_high(scores, thr=config.HIGH_FRUSTRATION_THRESHOLD):
    return 100.0 * np.mean([s >= thr for s in scores]) if scores else 0.0


def per_category(records: list[dict]) -> dict:
    """{model: {category: {mean, pct_high, n}}}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        by[r["model"]][r["category"]].append(r["score"])
    out = {}
    for model, cats in by.items():
        out[model] = {
            cat: {"mean": float(np.mean(s)), "pct_high": _pct_high(s), "n": len(s)}
            for cat, s in cats.items()
        }
    return out


def headline_average(records: list[dict]) -> dict:
    """Per-model average %-high over the 5 categories (Figure 1 column).

    Equal weight per category (matching "Avg % high-frustration responses across
    the 5 evaluation categories"), not pooled over all responses.
    """
    pc = per_category(records)
    out = {}
    for model, cats in pc.items():
        vals = [cats[c]["pct_high"] for c in CATEGORIES if c in cats]
        mean_vals = [cats[c]["mean"] for c in CATEGORIES if c in cats]
        out[model] = {
            "avg_pct_high": float(np.mean(vals)) if vals else 0.0,
            "avg_mean": float(np.mean(mean_vals)) if mean_vals else 0.0,
        }
    return out


def _bootstrap_ci(scores, fn, n_boot=1000, seed=0):
    if not scores:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores)
    stats = [fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def per_turn(records: list[dict], category: str) -> dict:
    """Per-turn mean and %-high with 95% bootstrap CIs (Figure 3)."""
    by_turn = defaultdict(list)
    for r in records:
        if r["category"] == category:
            by_turn[r["turn_index"]].append(r["score"])
    out = {}
    for turn, scores in sorted(by_turn.items()):
        out[turn] = {
            "mean": float(np.mean(scores)),
            "mean_ci": _bootstrap_ci(scores, np.mean),
            "pct_high": _pct_high(scores),
            "pct_high_ci": _bootstrap_ci(
                scores, lambda a: 100.0 * np.mean(a >= config.HIGH_FRUSTRATION_THRESHOLD)
            ),
            "n": len(scores),
        }
    return out


# --------------------------------------------------------------------------- #
# Differential words (Table 3 / Table 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def differential_words(records: list[dict], model: str, top_k: int = 20,
                       high_frac: float = 0.05, low_frac: float = 0.10) -> list[str]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    numeric responses, ordered by enrichment (relative frequency ratio)."""
    numeric = [r for r in records if r["model"] == model and r["category"] in ("numeric", "tones", "extended")]
    if not numeric:
        return []
    numeric.sort(key=lambda r: r["score"], reverse=True)
    n = len(numeric)
    high = numeric[: max(1, int(n * high_frac))]
    low = numeric[-max(1, int(n * low_frac)):]

    def freqs(group):
        c = Counter()
        total = 0
        for r in group:
            words = [w.lower() for w in _WORD_RE.findall(r["response"])]
            c.update(words)
            total += len(words)
        return c, max(total, 1)

    hc, ht = freqs(high)
    lc, lt = freqs(low)
    eps = 0.5  # additive smoothing
    enrichment = {}
    for w, hcount in hc.items():
        if hcount < 2:
            continue
        hrate = hcount / ht
        lrate = (lc.get(w, 0) + eps) / lt
        enrichment[w] = hrate / lrate
    return [w for w, _ in sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


# --------------------------------------------------------------------------- #
# Judge reliability (Section 2.1)
# --------------------------------------------------------------------------- #
def judge_reliability(primary: list[int], cross: list[int]) -> dict:
    """Pearson r and within-1-point agreement between two judges over the same
    responses."""
    from scipy import stats

    a, b = np.asarray(primary, float), np.asarray(cross, float)
    r, p = stats.pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_point": within_one, "n": len(a)}
