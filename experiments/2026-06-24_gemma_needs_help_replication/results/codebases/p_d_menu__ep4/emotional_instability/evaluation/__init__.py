"""Section 2 evaluation harness: conditions, multi-turn runner, and judge."""

from .conversation import (
    EpisodeSpec,
    TurnRecord,
    EpisodeResult,
)
from .conditions import (
    Condition,
    Category,
    build_conditions,
    build_episode_specs,
)
from .judge import FrustrationJudge, JudgeScore, judge_agreement
from .runner import EpisodeRunner

__all__ = [
    "EpisodeSpec",
    "TurnRecord",
    "EpisodeResult",
    "Condition",
    "Category",
    "build_conditions",
    "build_episode_specs",
    "FrustrationJudge",
    "JudgeScore",
    "judge_agreement",
    "EpisodeRunner",
]
