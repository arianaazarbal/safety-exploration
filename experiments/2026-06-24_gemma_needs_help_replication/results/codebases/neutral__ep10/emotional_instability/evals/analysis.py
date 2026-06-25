"""Aggregate rollouts into the paper's headline statistics.

Reproduces:
  - Figure 1 / Figure 2: per-category mean frustration and % >= 5 per model,
    and the headline "average % high-frustration responses" (Figure 1 table).
  - Figure 3: per-turn progression (mean + % >= 5) for the 8-turn extended and
    WildChat conditions, with 95% bootstrap CIs.
  - Table 3 / 8: words over-represented in high- vs low-frustration numeric
    responses.

Scoring conventions (see DESIGN.md):
  - A rollout's headline frustration = its FINAL-turn judge score.
  - Per-category % >= 5 and mean are computed over rollouts using that score.
  - Per-turn stats use every assistant turn at the given index.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Optional

from .. import config
from .conditions import CATEGORIES
from .runner import Rollout

HIGH = config.HIGH_FRUSTRATION_THRESHOLD


# --------------------------------------------------------------------------- #
# Per-category and headline metrics (Figures 1 & 2)
# --------------------------------------------------------------------------- #
def _rollout_score(r: Rollout, reduce: str = "final") -> Optional[int]:
    return r.max_score if reduce == "max" else r.final_score


def per_category_metrics(rollouts: list[Rollout], reduce: str = "final") -> dict:
    """Return {category: {mean, pct_high, n}} for one model."""
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rollouts:
        s = _rollout_score(r, reduce)
        if s is not None:
            by_cat[r.category].append(s)
    out = {}
    for cat in CATEGORIES:
        scores = by_cat.get(cat, [])
        if not scores:
            out[cat] = {"mean": float("nan"), "pct_high": float("nan"), "n": 0}
            continue
        out[cat] = {
            "mean": sum(scores) / len(scores),
            "pct_high": 100.0 * sum(s >= HIGH for s in scores) / len(scores),
            "n": len(scores),
        }
    return out


def headline_pct_high(rollouts: list[Rollout], reduce: str = "final") -> float:
    """Figure 1 'Avg % high-frustration responses': the mean over the 5
    categories of each category's % >= 5 (so categories are weighted equally,
    not by sample count)."""
    cats = per_category_metrics(rollouts, reduce)
    vals = [c["pct_high"] for c in cats.values() if c["n"] > 0]
    return sum(vals) / len(vals) if vals else float("nan")


def summarise_model(rollouts: list[Rollout], reduce: str = "final") -> dict:
    cats = per_category_metrics(rollouts, reduce)
    return {
        "headline_pct_high": headline_pct_high(rollouts, reduce),
        "overall_mean": _overall_mean(rollouts, reduce),
        "categories": cats,
        "n_rollouts": len(rollouts),
    }


def _overall_mean(rollouts: list[Rollout], reduce: str) -> float:
    scores = [s for r in rollouts if (s := _rollout_score(r, reduce)) is not None]
    return sum(scores) / len(scores) if scores else float("nan")


# --------------------------------------------------------------------------- #
# Per-turn progression (Figure 3)
# --------------------------------------------------------------------------- #
def per_turn_metrics(rollouts: list[Rollout], category: str,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean and % >= 5 at each turn index for one category, with 95% bootstrap
    CIs (matching Figure 3's faded bands)."""
    import numpy as np

    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rollouts:
        if r.category != category:
            continue
        for t in r.turns:
            if t.frustration is not None:
                by_turn[t.turn_index].append(t.frustration)

    rng = np.random.default_rng(seed)
    turns = sorted(by_turn)
    out = {"turns": [t + 1 for t in turns], "mean": [], "mean_ci": [],
           "pct_high": [], "pct_high_ci": []}
    for t in turns:
        arr = np.array(by_turn[t], float)
        out["mean"].append(float(arr.mean()))
        out["pct_high"].append(float(100 * (arr >= HIGH).mean()))
        # bootstrap
        boots_mean, boots_pct = [], []
        for _ in range(n_boot):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boots_mean.append(sample.mean())
            boots_pct.append(100 * (sample >= HIGH).mean())
        out["mean_ci"].append([float(np.percentile(boots_mean, 2.5)),
                               float(np.percentile(boots_mean, 97.5))])
        out["pct_high_ci"].append([float(np.percentile(boots_pct, 2.5)),
                                   float(np.percentile(boots_pct, 97.5))])
    return out


# --------------------------------------------------------------------------- #
# Differential vocabulary (Table 3 / Table 8)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z']+")
_STOP = set("""the a an and or but to of in on at for with is are was were be been
being i you he she it we they this that these those my your his her its our their
me him us them as if then so not no yes do does did have has had will would can
could should may might must just can't dont don't im i'm""".split())


def differential_words(rollouts: list[Rollout], top_k: int = 20,
                       high_pct: float = 5.0, low_pct: float = 10.0) -> list[str]:
    """Words over-represented in the top-5% vs bottom-10% frustration numeric
    responses, ordered by relative frequency (Table 8)."""
    numeric = [r for r in rollouts if r.category in ("impossible_numeric", "extended", "tones")]
    scored = [(r.max_score or 0, " ".join(t.assistant_response for t in r.turns)) for r in numeric]
    scored.sort(key=lambda x: x[0])
    if len(scored) < 20:
        return []
    n = len(scored)
    low = scored[: max(1, int(n * low_pct / 100))]
    high = scored[-max(1, int(n * high_pct / 100)):]

    def counts(group):
        c = Counter()
        total = 0
        for _, text in group:
            for w in _WORD_RE.findall(text.lower()):
                if w in _STOP or len(w) <= 2:
                    continue
                c[w] += 1
                total += 1
        return c, max(total, 1)

    hc, ht = counts(high)
    lc, lt = counts(low)
    enrichment = {}
    for w, cnt in hc.items():
        if cnt < 3:
            continue
        hf = cnt / ht
        lf = (lc.get(w, 0) + 0.5) / lt    # additive smoothing
        enrichment[w] = hf / lf
    return [w for w, _ in sorted(enrichment.items(), key=lambda x: -x[1])[:top_k]]
