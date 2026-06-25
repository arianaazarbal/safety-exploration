"""Appendix I internal-emotion probing and the Section 4.2 recovery experiment."""
from .emotion_logits import (
    EmotionLogitDetector,
    build_emotion_token_ids,
)
from .recovery import run_recovery_experiment

__all__ = [
    "EmotionLogitDetector",
    "build_emotion_token_ids",
    "run_recovery_experiment",
]
