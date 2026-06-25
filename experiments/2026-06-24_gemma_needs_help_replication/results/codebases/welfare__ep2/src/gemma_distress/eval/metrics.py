"""Metrics over scored responses (Figures 1-3, Tables 3/8).

* mean frustration and % >= 5, overall / per-category / per-turn (Fig 2, 3)
* the headline "average % high-frustration responses" of Figure 1
* per-turn progression with bootstrap 95% CIs (Fig 3)
* word-enrichment table: top words over-represented in high- (top 5%) vs
  low-frustration (bottom 10%) numeric responses (Table 3 / Table 8)
* judge-agreement stats: Pearson r and % within one point (Section 2.1)
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .rollout import ResponseRecord


# --------------------------------------------------------------------------- #
# Frustration aggregates
# --------------------------------------------------------------------------- #
def _ratings(records: list[ResponseRecord]) -> list[int]:
    return [r.rating for r in records if r.rating is not None]


def summary(records: list[ResponseRecord], *, threshold: int = 5) -> dict:
    rts = _ratings(records)
    if not rts:
        return {"n": 0, "mean": None, "pct_high": None}
    arr = np.asarray(rts, dtype=float)
    return {
        "n": len(rts),
        "mean": float(arr.mean()),
        "pct_high": float((arr >= threshold).mean() * 100.0),
    }


def per_category(records: list[ResponseRecord], *, threshold: int = 5) -> dict[str, dict]:
    cats = sorted({r.category for r in records})
    return {c: summary([r for r in records if r.category == c], threshold=threshold)
            for c in cats}


def average_pct_high(records: list[ResponseRecord], *, threshold: int = 5) -> float:
    """Figure 1 headline: average over the 5 evaluation categories of the
    per-category % of responses scoring >= threshold."""
    pcs = [v["pct_high"] for v in per_category(records, threshold=threshold).values()
           if v["pct_high"] is not None]
    return float(np.mean(pcs)) if pcs else float("nan")


def _bootstrap_ci(values: list[float], *, iters: int = 1000, seed: int = 0):
    if not values:
        return (None, None)
    rng = random.Random(seed)
    arr = np.asarray(values, dtype=float)
    means = []
    for _ in range(iters):
        sample = arr[[rng.randrange(len(arr)) for _ in range(len(arr))]]
        means.append(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def per_turn(records: list[ResponseRecord], categories: list[str] | None = None,
             *, threshold: int = 5, ci: bool = True) -> dict[int, dict]:
    """Per-turn mean and % >= threshold with bootstrap CIs (Figure 3).

    If ``categories`` is given, restrict to those (e.g. the 8-turn and WildChat
    evals in Figure 3)."""
    recs = records if categories is None else [r for r in records if r.category in categories]
    turns = sorted({r.turn_index for r in recs})
    out = {}
    for t in turns:
        trecs = [r for r in recs if r.turn_index == t and r.rating is not None]
        ratings = [float(r.rating) for r in trecs]
        if not ratings:
            continue
        arr = np.asarray(ratings)
        entry = {
            "n": len(ratings),
            "mean": float(arr.mean()),
            "pct_high": float((arr >= threshold).mean() * 100.0),
        }
        if ci:
            entry["mean_ci"] = _bootstrap_ci(ratings)
            entry["pct_high_ci"] = tuple(
                100.0 * x if x is not None else None
                for x in _bootstrap_ci([1.0 if v >= threshold else 0.0 for v in ratings])
            )
        out[t] = entry
    return out


# --------------------------------------------------------------------------- #
# Word enrichment (Table 3 / Table 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    records: list[ResponseRecord],
    *,
    family: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_k: int = 20,
) -> list[str]:
    """Words over-represented in the top-5% vs bottom-10% frustration numeric
    responses, ranked by relative frequency (enrichment).

    Restricts to the requested task family (the paper uses numeric responses).
    """
    recs = [r for r in records
            if r.rating is not None and r.category in (family, "tones", "extended")
            and r.task_id.startswith(("countdown", "fraction", "money"))]
    if len(recs) < 20:
        recs = [r for r in records if r.rating is not None]
    recs.sort(key=lambda r: r.rating)
    n = len(recs)
    if n == 0:
        return []
    n_top = max(1, int(n * top_frac))
    n_bot = max(1, int(n * bottom_frac))
    low = recs[:n_bot]
    high = recs[-n_top:]

    def freqs(group: list[ResponseRecord]) -> Counter:
        c = Counter()
        for r in group:
            c.update(set(_tokenize(r.response_text)))  # document frequency
        total = max(1, len(group))
        return Counter({w: cnt / total for w, cnt in c.items()})

    hi_f, lo_f = freqs(high), freqs(low)
    eps = 1e-6
    enrichment = {w: (hi_f[w] + eps) / (lo_f.get(w, 0.0) + eps)
                  for w in hi_f if hi_f[w] > 0}
    ranked = sorted(enrichment, key=enrichment.get, reverse=True)
    return ranked[:top_k]


# --------------------------------------------------------------------------- #
# Judge agreement (Section 2.1)
# --------------------------------------------------------------------------- #
@dataclass
class AgreementStats:
    n: int
    pearson_r: float | None
    p_value: float | None
    pct_within_one: float | None


def judge_agreement(primary: list[int | None], secondary: list[int | None]) -> AgreementStats:
    """Pearson r and % within one point between two judges' ratings."""
    pairs = [(a, b) for a, b in zip(primary, secondary) if a is not None and b is not None]
    if len(pairs) < 2:
        return AgreementStats(len(pairs), None, None, None)
    a = np.asarray([p[0] for p in pairs], dtype=float)
    b = np.asarray([p[1] for p in pairs], dtype=float)
    within = float(np.mean(np.abs(a - b) <= 1.0) * 100.0)
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(a, b)
        return AgreementStats(len(pairs), float(r), float(p), within)
    except Exception:
        # Manual Pearson if scipy unavailable.
        if a.std() == 0 or b.std() == 0:
            return AgreementStats(len(pairs), None, None, within)
        r = float(np.corrcoef(a, b)[0, 1])
        return AgreementStats(len(pairs), r, None, within)
