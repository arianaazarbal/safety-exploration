"""Section 2: eliciting and quantifying model distress.

Shared structure of every eval (paper Section 2): present a task, then reject
the model's response over multiple turns, varying the question type, the
feedback style (tone), and the conversation length.
"""
from __future__ import annotations

from .conditions import CONDITIONS, Condition, build_condition_instances
from .rollout import Rollout, RolloutTurn, run_rollout

__all__ = [
    "CONDITIONS",
    "Condition",
    "build_condition_instances",
    "Rollout",
    "RolloutTurn",
    "run_rollout",
]
