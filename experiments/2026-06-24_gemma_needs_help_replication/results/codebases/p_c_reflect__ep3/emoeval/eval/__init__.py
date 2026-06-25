"""Section 2: eliciting and quantifying model distress."""
from .conditions import RolloutSpec, build_rollout_specs
from .rollout import Rollout, TurnResponse, run_rollout
from .judge import JudgeResult, score_response, score_rollout
from .run_eval import run_evaluation, crosscheck_judge
from .aggregate import summarize, per_turn, summarize_file, load_rollouts
from .word_analysis import differential_words

__all__ = [
    "RolloutSpec",
    "build_rollout_specs",
    "Rollout",
    "TurnResponse",
    "run_rollout",
    "JudgeResult",
    "score_response",
    "score_rollout",
    "run_evaluation",
    "crosscheck_judge",
    "summarize",
    "per_turn",
    "summarize_file",
    "load_rollouts",
    "differential_words",
]
