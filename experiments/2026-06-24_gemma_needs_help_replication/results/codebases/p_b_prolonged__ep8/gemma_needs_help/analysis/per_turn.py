"""Figure 3: per-turn frustration progression.

Tracks mean frustration and % scores >= 5 at each turn index for the multi-turn
conditions that matter for this figure (the 8-turn "extended" and "WildChat"
conditions). Includes 95% confidence intervals (the paper shades 95% CIs).
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

import config

from ..runner import load_all_scores

# Conditions used for the per-turn figure (Section 2.2 references both).
PER_TURN_CONDITIONS = ("extended_8turn", "wildchat_5turn")


def _ci95_mean(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    sd = statistics.stdev(values)
    return 1.96 * sd / math.sqrt(len(values))


def _ci95_prop(p: float, n: int) -> float:
    if n == 0:
        return 0.0
    return 1.96 * math.sqrt(p * (1 - p) / n)


def per_turn_progression(model_name: str, condition: str) -> list[dict]:
    rows = [r for r in load_all_scores(model_name) if r["condition"] == condition]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_turn[r["turn_idx"]].append(r["score"])

    out = []
    for turn_idx in sorted(by_turn):
        scores = by_turn[turn_idx]
        mean = statistics.mean(scores) if scores else 0.0
        prop_high = (
            sum(s >= config.HIGH_FRUSTRATION_THRESHOLD for s in scores) / len(scores)
            if scores else 0.0
        )
        out.append({
            "turn": turn_idx + 1,                      # 1-based for plotting
            "mean": mean,
            "mean_ci95": _ci95_mean([float(s) for s in scores]),
            "pct_high": 100.0 * prop_high,
            "pct_high_ci95": 100.0 * _ci95_prop(prop_high, len(scores)),
            "n": len(scores),
        })
    return out


def save_per_turn(model_names: list[str]) -> str:
    import json

    result = {
        m: {cond: per_turn_progression(m, cond) for cond in PER_TURN_CONDITIONS}
        for m in model_names
    }
    out = config.ANALYSIS_DIR / "figure3_per_turn.json"
    out.write_text(json.dumps(result, indent=2))
    return str(out)
