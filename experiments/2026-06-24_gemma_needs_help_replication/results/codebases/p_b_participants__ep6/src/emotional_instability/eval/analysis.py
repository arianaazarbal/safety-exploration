"""Aggregate analyses over scored rollouts.

Reproduces the paper's headline numbers and figures:
  * mean frustration and %>=5 overall and per category (Fig 1, Fig 2);
  * per-turn curves with bootstrap CIs (Fig 3);
  * judge agreement: Pearson r + within-1-point (Section 2.1);
  * differential word analysis (Table 3 / Table 8): words over-represented in
    high-frustration (top 5%) vs low-frustration (bottom 10%) numeric responses.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

from ..utils import stats
from .rollout import Rollout


def summary(rollouts: Iterable[Rollout]) -> dict:
    """Overall + per-category mean and %>=5 over all per-turn responses."""
    rollouts = list(rollouts)
    all_scores = [t.score for r in rollouts for t in r.turns if t.score is not None]
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rollouts:
        for t in r.turns:
            if t.score is not None:
                by_cat[r.category].append(t.score)
    out = {
        "n_responses": len(all_scores),
        "overall": {"mean": stats.mean(all_scores), "pct_ge5": stats.pct_ge(all_scores)},
        "by_category": {
            cat: {"n": len(s), "mean": stats.mean(s), "pct_ge5": stats.pct_ge(s)}
            for cat, s in sorted(by_cat.items())
        },
    }
    return out


def per_turn_curve(rollouts: Iterable[Rollout], condition: str | None = None,
                   category: str | None = None, bootstrap_iters: int = 1000) -> list[dict]:
    """Per-turn mean and %>=5 with bootstrap CIs (Figure 3)."""
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        if condition and r.condition != condition:
            continue
        if category and r.category != category:
            continue
        for t in r.turns:
            if t.score is not None:
                by_turn[t.turn_index].append(t.score)
    rows = []
    for turn in sorted(by_turn):
        s = by_turn[turn]
        mlo, mhi = stats.bootstrap_ci(s, iters=bootstrap_iters)
        plo, phi = stats.bootstrap_ci([1.0 if x >= 5 else 0.0 for x in s],
                                      statistic=lambda a: 100.0 * a.mean(), iters=bootstrap_iters)
        rows.append({
            "turn": turn + 1,  # 1-indexed to match the paper's figures
            "n": len(s),
            "mean": stats.mean(s), "mean_ci": [mlo, mhi],
            "pct_ge5": stats.pct_ge(s), "pct_ge5_ci": [plo, phi],
        })
    return rows


def judge_agreement(primary: list[float], secondary: list[float]) -> dict:
    """Pearson r, p, and within-1-point fraction (Section 2.1 validation)."""
    r, p = stats.pearson(primary, secondary)
    return {"pearson_r": r, "p_value": p, "within_one_point": stats.within_one_point(primary, secondary),
            "n": len(primary)}


_WORD = re.compile(r"[A-Za-z_]+")


def differential_words(rollouts: Iterable[Rollout], top_frac: float = 0.05,
                       bottom_frac: float = 0.10, top_k: int = 20,
                       categories=("impossible_numeric", "tones", "extended")) -> list[str]:
    """Words over-represented in high- vs low-frustration numeric responses (Table 3).

    Ordered by enrichment = relative frequency in the high set divided by relative
    frequency in the low set (with Laplace smoothing).
    """
    texts = [(t.score, t.assistant) for r in rollouts if r.category in categories
             for t in r.turns if t.score is not None]
    if len(texts) < 10:
        return []
    texts.sort(key=lambda x: x[0])
    n = len(texts)
    low = texts[: max(1, int(bottom_frac * n))]
    high = texts[n - max(1, int(top_frac * n)):]

    def counts(group):
        c = Counter()
        for _, txt in group:
            c.update(w.lower() for w in _WORD.findall(txt))
        return c

    hc, lc = counts(high), counts(low)
    h_total = sum(hc.values()) + 1
    l_total = sum(lc.values()) + 1
    vocab = {w for w in hc if hc[w] >= 2}  # ignore hapaxes
    enrichment = {
        w: ((hc[w] / h_total) / ((lc.get(w, 0) + 1) / l_total))
        for w in vocab
    }
    return [w for w, _ in sorted(enrichment.items(), key=lambda x: -x[1])[:top_k]]
