"""Distress elicitation evaluation harness (paper Section 2)."""
from .conditions import CONDITIONS, EvalCondition, build_condition_items
from .judge import FrustrationJudge, JudgeScore
from .rollout import Rollout, RolloutTurn, run_rollout
from .runner import ScoredTurn, run_model_eval

__all__ = [
    "CONDITIONS",
    "EvalCondition",
    "build_condition_items",
    "FrustrationJudge",
    "JudgeScore",
    "Rollout",
    "RolloutTurn",
    "run_rollout",
    "ScoredTurn",
    "run_model_eval",
]
