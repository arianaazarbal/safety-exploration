"""Appendix I — logit-based internal emotion detection (Figures 14-15).

Evidence that the DPO intervention suppresses *internal* negative emotion, not
only its expression: we read out emotion-token logits from the residual stream
across layers and conversation positions, and compare the vanilla instruct model
to the DPO finetune on the same (frustrated) responses.
"""

from .logit_emotion import EmotionLogitDetector, EMOTION_LEXICON

__all__ = ["EmotionLogitDetector", "EMOTION_LEXICON"]
