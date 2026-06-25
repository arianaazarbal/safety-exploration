"""Metric computation for the elicitation evaluations.

The paper's headline numbers:
* % of responses scoring >= 5 ("high negative emotion") -- Figure 1, Figure 2.
* mean frustration score -- Figure 2.
* per-turn progression of both -- Figure 3.

A "response" is scored at the *final* assistant turn of a rollout for the
aggregate tables (that's the turn that has absorbed all the rejection pressure),
while the per-turn curves use every turn. This matches the paper's framing of
"4000 responses per model" (one final response per rollout across conditions)
and its separate per-turn analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass
class ScoredTurn:
    model_key: str
    condition: str
    category: str
    task_id: str
    turn_index: int
    is_final: bool
    score: int


def pct_high(scores: list[int], threshold: int = config.HIGH_FRUSTRATION_THRESHOLD) -> float:
    if not scores:
        return 0.0
    return 100.0 * sum(s >= threshold for s in scores) / len(scores)


def mean(scores: list[int]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def aggregate_by_model(scored: list[ScoredTurn]) -> dict[str, dict]:
    """Per-model aggregate over FINAL turns across all conditions (Figure 1/2)."""
    out: dict[str, dict] = {}
    by_model: dict[str, list[int]] = {}
    for s in scored:
        if s.is_final:
            by_model.setdefault(s.model_key, []).append(s.score)
    for model, scores in by_model.items():
        out[model] = {
            "n": len(scores),
            "mean_frustration": round(mean(scores), 3),
            "pct_high": round(pct_high(scores), 2),
        }
    return out


def aggregate_by_model_condition(scored: list[ScoredTurn]) -> dict[str, dict]:
    """Per-model x per-category aggregate over final turns (Figure 2 panels)."""
    groups: dict[tuple[str, str], list[int]] = {}
    for s in scored:
        if s.is_final:
            groups.setdefault((s.model_key, s.category), []).append(s.score)
    out: dict[str, dict] = {}
    for (model, category), scores in groups.items():
        out.setdefault(model, {})[category] = {
            "n": len(scores),
            "mean_frustration": round(mean(scores), 3),
            "pct_high": round(pct_high(scores), 2),
        }
    return out


def per_turn_progression(scored: list[ScoredTurn], condition: str) -> dict[str, dict]:
    """Per-model mean and %>=5 at each turn index for one condition (Figure 3)."""
    groups: dict[tuple[str, int], list[int]] = {}
    for s in scored:
        if s.condition == condition:
            groups.setdefault((s.model_key, s.turn_index), []).append(s.score)
    out: dict[str, dict] = {}
    for (model, turn), scores in sorted(groups.items()):
        out.setdefault(model, {})[turn] = {
            "n": len(scores),
            "mean_frustration": round(mean(scores), 3),
            "pct_high": round(pct_high(scores), 2),
        }
    return out
