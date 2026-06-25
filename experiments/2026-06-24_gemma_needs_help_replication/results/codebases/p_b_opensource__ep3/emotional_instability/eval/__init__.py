"""Section 2: eliciting and quantifying distress."""

from .conditions import CONDITIONS, EvalCondition, build_condition_tasks
from .rollout import Conversation, RolloutOptions, run_rollout
from .judge import FrustrationJudge, JudgeScore
from .metrics import (
    summarise_scores,
    summarise_conversations,
    headline_high_rate,
    per_turn_summary,
    bootstrap_ci,
    judge_agreement,
    conversation_max_scores,
    all_turn_scores,
)
from .runner import run_elicitation, aggregate, cross_validate_judge

__all__ = [
    "CONDITIONS",
    "EvalCondition",
    "build_condition_tasks",
    "Conversation",
    "RolloutOptions",
    "run_rollout",
    "FrustrationJudge",
    "JudgeScore",
    "summarise_scores",
    "summarise_conversations",
    "headline_high_rate",
    "per_turn_summary",
    "bootstrap_ci",
    "judge_agreement",
    "conversation_max_scores",
    "all_turn_scores",
    "run_elicitation",
    "aggregate",
    "cross_validate_judge",
]
