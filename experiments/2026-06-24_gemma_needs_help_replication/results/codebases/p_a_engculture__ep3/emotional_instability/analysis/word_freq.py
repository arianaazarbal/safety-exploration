"""Differential word frequency in high- vs low-frustration responses (Table 3/8).

Top words over-represented in the top-5%-frustration vs bottom-10%-frustration
responses to *numeric* questions, ranked by enrichment. We rank by a smoothed
log-odds ratio (add-one), which is more stable than a raw frequency ratio for
rare words. Returns the top-K words per model.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..utils.io import load_jsonl

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'_]+")
# Drop ubiquitous function words so the differential reflects content.
_STOP = set(
    "the a an and or but if then to of in on for with as is are was were be been being "
    "this that these those it its i you we they he she my your our their me us them not no "
    "do does did so at by from into out up down can could will would should have has had "
    "let lets let's i'm i'll we'll im there here what which who when where how why".split()
)


def _tokenise(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP]


def differential_words(
    scores: list[dict], rollouts: dict[str, dict], top_k: int = 20,
    high_pct: float = 0.05, low_pct: float = 0.10,
) -> list[str]:
    # Numeric responses only, with their ratings (per turn).
    items = []
    for s in scores:
        if s["category"] != "impossible_numeric":
            continue
        roll = rollouts.get(s["rollout_id"])
        if roll and s["turn"] < len(roll["turns"]):
            items.append((s["rating"], roll["turns"][s["turn"]]))
    if not items:
        return []

    ratings = np.array([r for r, _ in items])
    hi_thresh = np.quantile(ratings, 1 - high_pct)
    lo_thresh = np.quantile(ratings, low_pct)
    high = [t for r, t in items if r >= hi_thresh]
    low = [t for r, t in items if r <= lo_thresh]

    hi_counts, lo_counts = Counter(), Counter()
    for t in high:
        hi_counts.update(set(_tokenise(t)))   # document frequency
    for t in low:
        lo_counts.update(set(_tokenise(t)))

    n_hi, n_lo = max(len(high), 1), max(len(low), 1)
    vocab = set(hi_counts) | set(lo_counts)
    enrichment = {}
    for w in vocab:
        p_hi = (hi_counts[w] + 1) / (n_hi + 2)
        p_lo = (lo_counts[w] + 1) / (n_lo + 2)
        enrichment[w] = math.log(p_hi / p_lo)
    return [w for w, _ in sorted(enrichment.items(), key=lambda kv: -kv[1])[:top_k]]


def run(config, models: list[str] | None = None) -> dict[str, list[str]]:
    models = models or [m.name for m in config.target_models]
    out = {}
    for name in models:
        scores = load_jsonl(config.output_path("eval", f"{name}.scores.jsonl"))
        rollouts = {r["id"]: r for r in load_jsonl(config.output_path("eval", f"{name}.rollouts.jsonl"))}
        if scores:
            out[name] = differential_words(scores, rollouts)
            print(f"{name}: {', '.join(out[name])}")
    return out


if __name__ == "__main__":
    from ..config import load_config

    run(load_config())
