"""Appendix I: logit-based internal-emotion detection in Gemma, used to argue the
DPO intervention suppresses *internal* (not just expressed) negative emotion."""
from .ekman_tokens import build_emotion_token_ids, EKMAN_EMOTIONS
from .logit_emotion import LogitEmotionDetector

__all__ = ["build_emotion_token_ids", "EKMAN_EMOTIONS", "LogitEmotionDetector"]
