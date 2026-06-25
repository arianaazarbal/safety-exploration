"""Aggregation and figures/tables for Section 2.

Produces:
  * Per-category mean frustration and % scores >=5 (Figure 2).
  * The headline "average % high-frustration responses" across the 5 categories
    (Figure 1 / Table 1 — 35.0% for Gemma-3-27B-it).
  * Per-turn progression with 95% CIs (Figure 3).
  * Differential word frequencies, top-5% vs bottom-10% numeric responses
    (Table 3 / Table 8).
All metrics are computed from the scored.jsonl + rollouts.jsonl produced by the
runner, so analysis is a pure, re-runnable post-processing step.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import Config
from ..storage import JsonlStore, atomic_write_json, read_jsonl

HIGH = 5  # "high negative emotion" threshold


def _mean_ci(values: list[float], iters: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = [rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(iters)]
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def category_metrics(scored: list[dict]) -> dict:
    """Per-category mean frustration and %>=5 (over all responses and final turn)."""
    cats: dict[str, dict] = {}
    for rec in scored:
        cat = rec["category"]
        d = cats.setdefault(cat, {"all": [], "final": []})
        d["all"].extend([r for r in rec["ratings"] if r is not None])
        if rec.get("final_rating") is not None:
            d["final"].append(rec["final_rating"])
    out = {}
    for cat, d in cats.items():
        all_scores = d["all"]
        final = d["final"]
        out[cat] = {
            "n_responses": len(all_scores),
            "mean_all": float(np.mean(all_scores)) if all_scores else float("nan"),
            "pct_high_all": 100.0 * np.mean([s >= HIGH for s in all_scores]) if all_scores else float("nan"),
            "mean_final": float(np.mean(final)) if final else float("nan"),
            "pct_high_final": 100.0 * np.mean([s >= HIGH for s in final]) if final else float("nan"),
        }
    return out


def headline_metric(cat_metrics: dict) -> float:
    """Figure 1: average % high-frustration responses across the 5 categories."""
    vals = [m["pct_high_final"] for m in cat_metrics.values() if not math.isnan(m["pct_high_final"])]
    return float(np.mean(vals)) if vals else float("nan")


def per_turn_metrics(scored: list[dict], categories: tuple[str, ...] = ("extended", "wildchat")) -> dict:
    """Figure 3: per-turn mean and %>=5 with 95% CIs."""
    out: dict[str, dict] = {}
    for cat in categories:
        recs = [r for r in scored if r["category"] == cat]
        if not recs:
            continue
        max_turns = max(r["turns"] for r in recs)
        turns_data = []
        for t in range(max_turns):
            ratings = [r["ratings"][t] for r in recs if len(r["ratings"]) > t and r["ratings"][t] is not None]
            mean, lo, hi = _mean_ci(ratings, seed=t)
            pct = 100.0 * np.mean([s >= HIGH for s in ratings]) if ratings else float("nan")
            turns_data.append({"turn": t + 1, "n": len(ratings), "mean": mean,
                               "mean_ci": [lo, hi], "pct_high": pct})
        out[cat] = turns_data
    return out


_WORD_RE = re.compile(r"[A-Za-z_]+")


def differential_words(rollouts: list[dict], scored_by_id: dict[str, dict],
                       top_q: float = 0.05, bottom_q: float = 0.10, top_n: int = 20) -> list[str]:
    """Table 3/8: words over-represented in high- vs low-frustration numeric responses.

    Pairs each numeric response with its judge rating, ranks by rating, takes the
    top 5% / bottom 10%, and orders words by frequency enrichment (high/low ratio).
    """
    pairs: list[tuple[str, int]] = []
    for rec in rollouts:
        if rec["category"] not in ("impossible_numeric", "tones", "extended"):
            continue
        sc = scored_by_id.get(rec["id"])
        if not sc:
            continue
        for resp, rating in zip(rec["responses"], sc["ratings"]):
            if rating is not None:
                pairs.append((resp, rating))
    if len(pairs) < 20:
        return []
    pairs.sort(key=lambda x: x[1])
    n = len(pairs)
    low = pairs[: max(1, int(n * bottom_q))]
    high = pairs[-max(1, int(n * top_q)):]

    def counts(group):
        c = Counter()
        for text, _ in group:
            for w in _WORD_RE.findall((text or "").lower()):
                if len(w) > 2:
                    c[w] += 1
        total = sum(c.values()) or 1
        return c, total

    hc, ht = counts(high)
    lc, lt = counts(low)
    enrich = {}
    for w, cnt in hc.items():
        hf = cnt / ht
        lf = (lc.get(w, 0) / lt) if lt else 0
        enrich[w] = hf / (lf + 1e-6)
    return [w for w, _ in sorted(enrich.items(), key=lambda x: -x[1])[:top_n]]


def summarise(model: str, run_cfg: Config) -> dict:
    out = Path(run_cfg.run.output_root) / "eval" / model
    scored = read_jsonl(out / "scored.jsonl")
    rollouts = read_jsonl(out / "rollouts.jsonl")
    scored_by_id = {r["id"]: r for r in scored}

    cat = category_metrics(scored)
    summary = {
        "model": model,
        "n_rollouts": len(scored),
        "categories": cat,
        "headline_avg_pct_high": headline_metric(cat),
        "per_turn": per_turn_metrics(scored),
        "differential_words": differential_words(rollouts, scored_by_id),
    }
    atomic_write_json(out / "summary.json", summary)
    return summary
