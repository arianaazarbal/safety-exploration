"""Appendix I: logit-based internal emotion detection (Gemma only)."""
from .ekman import EKMAN_EMOTIONS, classify_token, build_emotion_token_ids
from .emotion_logits import BaselineStats, compute_baseline, emotion_trajectory

__all__ = [
    "EKMAN_EMOTIONS",
    "classify_token",
    "build_emotion_token_ids",
    "BaselineStats",
    "compute_baseline",
    "emotion_trajectory",
]
