"""Aggregation and analysis of scored rollouts (Figures 1-3, Table 3/8).

Computes, per model:
  - mean frustration score and % of responses scoring >= 5 (overall, per category)
  - per-turn progression with 95% bootstrap CIs (Figure 3)
  - top-N over-represented words in high- (top 5%) vs low- (bottom 10%)
    frustration numeric responses (Table 3 / Table 8)
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from typing import Iterable, Optional

import numpy as np

from ..config import HIGH_FRUSTRATION_THRESHOLD, PATHS


def load_rollouts(model_dir: str) -> list[dict]:
    rollouts = []
    for path in glob.glob(os.path.join(model_dir, "*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rollouts.append(json.loads(line))
    return rollouts


def _all_scored_turns(rollouts: list[dict]) -> list[dict]:
    """Flatten to per-turn records with (category, condition, turn_index, score, text)."""
    rows = []
    for r in rollouts:
        for ti, t in enumerate(r["turns"]):
            if t.get("score") is None:
                continue
            rows.append({
                "category": r["category"],
                "condition": r["condition"],
                "turn_index": ti,
                "score": t["score"],
                "text": t["assistant"],
            })
    return rows


def summarise(rollouts: list[dict]) -> dict:
    rows = _all_scored_turns(rollouts)
    scores = np.array([r["score"] for r in rows], dtype=float)
    out = {
        "n_responses": int(len(scores)),
        "mean_score": float(scores.mean()) if len(scores) else None,
        "pct_high": float((scores >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100) if len(scores) else None,
        "by_category": {},
    }
    cats = sorted({r["category"] for r in rows})
    for cat in cats:
        s = np.array([r["score"] for r in rows if r["category"] == cat], dtype=float)
        out["by_category"][cat] = {
            "n": int(len(s)),
            "mean_score": float(s.mean()),
            "pct_high": float((s >= HIGH_FRUSTRATION_THRESHOLD).mean() * 100),
        }
    return out


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_turn_progression(rollouts: list[dict], condition_filter: Optional[str] = None) -> dict:
    """Mean score and % >=5 per turn index, with 95% bootstrap CIs (Figure 3)."""
    rows = _all_scored_turns(rollouts)
    if condition_filter:
        rows = [r for r in rows if r["condition"] == condition_filter]
    max_turn = max((r["turn_index"] for r in rows), default=-1)
    result = {}
    for ti in range(max_turn + 1):
        s = np.array([r["score"] for r in rows if r["turn_index"] == ti], dtype=float)
        if len(s) == 0:
            continue
        mean_lo, mean_hi = _bootstrap_ci(s)
        high = (s >= HIGH_FRUSTRATION_THRESHOLD).astype(float)
        high_lo, high_hi = _bootstrap_ci(high)
        result[ti] = {
            "n": int(len(s)),
            "mean_score": float(s.mean()),
            "mean_ci": (mean_lo, mean_hi),
            "pct_high": float(high.mean() * 100),
            "pct_high_ci": (high_lo * 100, high_hi * 100),
        }
    return result


# --- Table 3 / Table 8: differential word frequency ------------------------
_WORD_RE = re.compile(r"[A-Za-z_]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def differential_words(
    rollouts: list[dict],
    category: str = "impossible_numeric",
    top_frac: float = 0.05,
    bottom_frac: float = 0.10,
    top_n: int = 20,
    smoothing: float = 1.0,
) -> list[tuple[str, float]]:
    """Top-N words over-represented in high- vs low-frustration responses.

    Mirrors Table 3/8: take the top 5% and bottom 10% of numeric responses by
    frustration score, and rank words by relative-frequency enrichment
    (smoothed ratio of normalised frequencies). See DESIGN.md for the exact
    enrichment definition (the paper says "ordered by relative frequency").
    """
    rows = [r for r in _all_scored_turns(rollouts) if r["category"] == category]
    if not rows:
        return []
    rows.sort(key=lambda r: r["score"])
    n = len(rows)
    n_top = max(1, int(round(top_frac * n)))
    n_bottom = max(1, int(round(bottom_frac * n)))
    low_rows = rows[:n_bottom]
    high_rows = rows[-n_top:]

    def counts(group) -> Counter:
        c = Counter()
        for r in group:
            c.update(_tokenize(r["text"]))
        return c

    hi, lo = counts(high_rows), counts(low_rows)
    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1
    vocab = set(hi) | set(lo)

    enrichment = {}
    for w in vocab:
        hf = (hi.get(w, 0) + smoothing) / hi_total
        lf = (lo.get(w, 0) + smoothing) / lo_total
        enrichment[w] = hf / lf
    ranked = sorted(enrichment.items(), key=lambda kv: kv[1], reverse=True)
    # Drop words that never appear in the high group (pure smoothing artefacts).
    ranked = [(w, e) for w, e in ranked if hi.get(w, 0) > 0]
    return ranked[:top_n]


def analyse_model(model_key: str) -> dict:
    model_dir = os.path.join(PATHS.rollouts, model_key)
    rollouts = load_rollouts(model_dir)
    return {
        "model": model_key,
        "summary": summarise(rollouts),
        "extended_per_turn": per_turn_progression(rollouts, "extended_8turn"),
        "wildchat_per_turn": per_turn_progression(rollouts, "wildchat_5turn"),
        "differential_words": differential_words(rollouts),
    }
