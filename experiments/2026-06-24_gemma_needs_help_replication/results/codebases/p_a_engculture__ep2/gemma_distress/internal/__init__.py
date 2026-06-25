"""Internal-emotion detection (Appendix I): Ekman lexicon + logit-lens detector."""

from .ekman import build_emotion_tokens, sample_random_tokens
from .logit_detector import InternalEmotionDetector

__all__ = ["build_emotion_tokens", "sample_random_tokens", "InternalEmotionDetector"]
