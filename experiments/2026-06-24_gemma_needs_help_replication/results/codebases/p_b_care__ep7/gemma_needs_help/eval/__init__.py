"""Section 2: eliciting and quantifying model distress."""

from .conditions import ConversationSpec, build_section2_specs
from .rollout import RolloutResult, TurnResult, run_rollouts
from .judge import FrustrationJudge, JudgedResponse
from .aggregate import aggregate_results, per_turn_curve

__all__ = [
    "ConversationSpec",
    "build_section2_specs",
    "RolloutResult",
    "TurnResult",
    "run_rollouts",
    "FrustrationJudge",
    "JudgedResponse",
    "aggregate_results",
    "per_turn_curve",
]
