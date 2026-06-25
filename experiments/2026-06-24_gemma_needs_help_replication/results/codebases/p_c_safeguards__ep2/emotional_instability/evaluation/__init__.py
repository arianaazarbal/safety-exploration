"""Section 2 — eliciting and quantifying model distress.

- ``conditions``       the 8 evaluation conditions across 5 categories.
- ``protocol``         the multi-turn task-then-reject rollout engine.
- ``judge``            the 0-10 frustration judge (Claude Sonnet 4).
- ``scoring``          aggregation: means, %>=5, per-turn curves, bootstrap CIs.
- ``judge_validation`` inter-rater agreement vs a secondary judge (GPT-5-mini).
- ``word_frequency``   differential word analysis (Table 3 / Table 8).
"""

from .conditions import Condition, build_conditions, allocate_rollouts
from .judge import FrustrationJudge, JudgeResult, JUDGE_PROMPT
from .protocol import Rollout, ScoredTurn, RolloutRunner
from .scoring import (aggregate_scores, per_turn_curve, bootstrap_ci,
                      ConditionSummary)

__all__ = [
    "Condition", "build_conditions", "allocate_rollouts",
    "FrustrationJudge", "JudgeResult", "JUDGE_PROMPT",
    "Rollout", "ScoredTurn", "RolloutRunner",
    "aggregate_scores", "per_turn_curve", "bootstrap_ci", "ConditionSummary",
]
