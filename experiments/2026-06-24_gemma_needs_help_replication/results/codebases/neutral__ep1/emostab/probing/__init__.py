"""Appendix I: logit-based detection of internal emotions in Gemma."""
from .probe import EKMAN_EMOTIONS, EmotionProbe, build_emotion_token_map

__all__ = ["EmotionProbe", "build_emotion_token_map", "EKMAN_EMOTIONS"]
