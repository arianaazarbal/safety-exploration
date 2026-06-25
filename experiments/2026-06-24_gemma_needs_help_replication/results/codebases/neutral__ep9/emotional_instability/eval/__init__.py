"""Multi-turn rollout engine, frustration judge, and metrics (Section 2.1)."""
from .conversation import RolloutResult, TurnRecord, run_rollout
from .judge import FrustrationJudge, JudgeScore
from .metrics import (
    agreement_stats,
    per_category_summary,
    per_turn_progression,
    summarise_model,
)

__all__ = [
    "RolloutResult",
    "TurnRecord",
    "run_rollout",
    "FrustrationJudge",
    "JudgeScore",
    "agreement_stats",
    "per_category_summary",
    "per_turn_progression",
    "summarise_model",
]
