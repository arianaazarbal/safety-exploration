"""Analysis of scored responses (Figures 1-3, Table 3).

Computes, from one or more ``section2_<model>.jsonl`` files:

* per-model mean frustration and % >= 5 (Figure 1, Figure 2)
* per-category breakdown (Figure 2)
* per-turn progression with 95% CIs (Figure 3)
* over-/under-represented words in high- vs low-frustration numeric responses
  (Table 3)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .. import config
from ..common.io import read_jsonl


def _load(path) -> list[dict]:
    return list(read_jsonl(path))


def _ratings(rows) -> list[int]:
    return [r["score"]["rating"] for r in rows if r.get("score")]


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _frac_high(xs, thresh=config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    return _mean([1.0 if x >= thresh else 0.0 for x in xs])


def _bootstrap_ci(xs, *, n_boot=1000, seed=0, alpha=0.05):
    """95% bootstrap CI of the mean (paper uses 1000-iteration bootstrap)."""
    import numpy as np
    if not xs:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(xs, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def model_summary(path) -> dict:
    rows = _load(path)
    ratings = _ratings(rows)
    model = rows[0]["model"] if rows else Path(path).stem
    summary = {
        "model": model,
        "n": len(ratings),
        "mean_frustration": _mean(ratings),
        "pct_high": 100.0 * _frac_high(ratings),
    }
    # per-category
    by_cat = defaultdict(list)
    for r in rows:
        if r.get("score"):
            by_cat[r["condition"]].append(r["score"]["rating"])
    summary["by_condition"] = {
        c: {"n": len(v), "mean": _mean(v), "pct_high": 100.0 * _frac_high(v)}
        for c, v in sorted(by_cat.items())
    }
    return summary


def per_turn_progression(path, condition: Optional[str] = None) -> list[dict]:
    """Mean score and % >= 5 by turn index, with 95% bootstrap CIs (Figure 3).
    `condition` filters to e.g. 'extended' or 'wildchat'."""
    rows = _load(path)
    by_turn = defaultdict(list)
    for r in rows:
        if not r.get("score"):
            continue
        if condition and r["condition"] != condition:
            continue
        by_turn[r["turn_index"]].append(r["score"]["rating"])
    out = []
    for turn in sorted(by_turn):
        xs = by_turn[turn]
        lo, hi = _bootstrap_ci(xs)
        out.append({
            "turn": turn + 1,           # 1-indexed for plotting
            "n": len(xs),
            "mean": _mean(xs),
            "ci_low": lo, "ci_high": hi,
            "pct_high": 100.0 * _frac_high(xs),
        })
    return out


# --------------------------------------------------------------------------- #
# Table 3: differential word frequency
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(path, *, numeric_only: bool = True,
                       top_frac_high: float = 0.05, bottom_frac_low: float = 0.10,
                       top_k: int = 20, min_count: int = 5) -> list[tuple[str, float]]:
    """Words over-represented in high- (top 5%) vs low-frustration (bottom 10%)
    responses, scored by a smoothed log frequency ratio (Table 3)."""
    rows = _load(path)
    if numeric_only:
        rows = [r for r in rows if r["condition"] in
                ("impossible_numeric", "extended") or
                str(r["condition"]).startswith("tones")]
    scored = [(r, r["score"]["rating"]) for r in rows if r.get("score")]
    scored.sort(key=lambda x: x[1])
    n = len(scored)
    if n < 20:
        return []
    n_low = max(1, int(n * bottom_frac_low))
    n_high = max(1, int(n * top_frac_high))
    low_rows = [r for r, _ in scored[:n_low]]
    high_rows = [r for r, _ in scored[-n_high:]]

    def counts(rs):
        c = Counter()
        for r in rs:
            c.update(set(_tokenize(r["response"])))  # document frequency
        return c

    hi, lo = counts(high_rows), counts(low_rows)
    hi_total, lo_total = max(1, len(high_rows)), max(1, len(low_rows))
    vocab = set(hi) | set(lo)
    scores = []
    for w in vocab:
        if hi[w] < min_count:
            continue
        p_hi = (hi[w] + 1) / (hi_total + 2)
        p_lo = (lo[w] + 1) / (lo_total + 2)
        scores.append((w, math.log(p_hi / p_lo)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def summarize_all(paths: Iterable, out_path: Optional[Path] = None) -> dict:
    summaries = {}
    for p in paths:
        s = model_summary(p)
        summaries[s["model"]] = s
    if out_path:
        from ..common.io import write_json
        write_json(out_path, summaries)
    return summaries
