"""Figure 3: per-turn frustration progression with 95% bootstrap CIs.

Reproduces the observation that Gemma-27B's mean frustration rises from ~1.5 to
~5.5 between turn 1 and turn 8, and that no model scores >=5 until turn 3 on
WildChat. Computed per (category, turn).
"""
from __future__ import annotations

from collections import defaultdict

from ..utils import read_jsonl
from ..utils.stats import mean_ci_bootstrap, frac_ge_threshold


def per_turn_curves(rollout_path: str, categories: list[str] | None = None,
                    threshold: int = 5, n_boot: int = 1000) -> dict:
    """Return {category: {turn: {mean, ci_lo, ci_hi, pct_ge, n}}}."""
    rows = read_jsonl(rollout_path)
    # category -> turn -> list of scores
    buckets: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rec in rows:
        if categories and rec["category"] not in categories:
            continue
        for turn in rec["turns"]:
            buckets[rec["category"]][turn["turn"]].append(turn["score"])

    out: dict[str, dict] = {}
    for cat, per_turn in buckets.items():
        out[cat] = {}
        for turn, scores in sorted(per_turn.items()):
            mean, lo, hi = mean_ci_bootstrap(scores, n_boot=n_boot)
            out[cat][turn] = {
                "mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "pct_ge_thresh": 100.0 * frac_ge_threshold(scores, threshold),
                "n": len(scores),
            }
    return out
