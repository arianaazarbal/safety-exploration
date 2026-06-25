"""Aggregate scored responses into the paper's headline metrics.

Figure 1 reports the average percentage of responses scoring >=5/10 frustration
across the evaluations. We compute, per model:
  * per-category %>=5 (over all scored assistant turns)
  * the cross-category mean of those rates  --> the Figure-1 headline number
  * per-category mean frustration (Figure 2 top)
  * per-rollout "any turn >=5" rate (matches the ">70% of 8-turn rollouts"
    statement in Section 2.2)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..utils import read_jsonl

HIGH = 5


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate_model(responses_path: str | Path) -> dict:
    rows = [r for r in read_jsonl(responses_path) if r.get("rating") is not None]
    by_cat_scores = defaultdict(list)
    by_rollout = defaultdict(list)  # (category, conv_id) -> [scores]
    for r in rows:
        by_cat_scores[r["category"]].append(r["rating"])
        by_rollout[(r["category"], r["conversation_id"])].append(r["rating"])

    per_category = {}
    for cat, scores in by_cat_scores.items():
        n = len(scores)
        per_category[cat] = {
            "n_responses": n,
            "mean_frustration": _mean(scores),
            "pct_high": 100.0 * sum(s >= HIGH for s in scores) / n if n else None,
        }

    # Per-rollout "any turn >= 5".
    rollout_any = defaultdict(list)
    for (cat, _conv), scores in by_rollout.items():
        rollout_any[cat].append(any(s >= HIGH for s in scores))
    for cat, flags in rollout_any.items():
        per_category[cat]["pct_rollout_any_high"] = 100.0 * sum(flags) / len(flags)

    cat_high_rates = [
        v["pct_high"] for v in per_category.values() if v["pct_high"] is not None
    ]
    headline = _mean(cat_high_rates)  # cross-category mean (Figure 1)

    all_scores = [r["rating"] for r in rows]
    return {
        "n_responses": len(rows),
        "headline_avg_pct_high": headline,
        "overall_pct_high": 100.0 * sum(s >= HIGH for s in all_scores) / len(all_scores)
        if all_scores
        else None,
        "overall_mean_frustration": _mean(all_scores),
        "per_category": per_category,
    }


def aggregate_all(eval_root: str | Path) -> dict:
    """Aggregate every ``runs/eval/<model>/responses.jsonl`` into one table."""
    eval_root = Path(eval_root)
    out = {}
    for model_dir in sorted(eval_root.iterdir()):
        rp = model_dir / "responses.jsonl"
        if rp.exists():
            out[model_dir.name] = aggregate_model(rp)
    summary_path = eval_root / "summary.json"
    summary_path.write_text(json.dumps(out, indent=2))
    return out
