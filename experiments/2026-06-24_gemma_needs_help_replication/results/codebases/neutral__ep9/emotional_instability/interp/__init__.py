"""Appendix I: logit-based internal emotion detection.

Provides evidence for whether the DPO finetune suppresses *internal* negative
emotion (in central layers) rather than merely masking expression at the output.
"""
from .emotion_dictionary import EKMAN_EMOTIONS, build_emotion_token_ids
from .logit_lens import EmotionDetector

__all__ = ["EKMAN_EMOTIONS", "build_emotion_token_ids", "EmotionDetector"]
