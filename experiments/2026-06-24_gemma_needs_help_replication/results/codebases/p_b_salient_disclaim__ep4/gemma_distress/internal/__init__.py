"""Appendix I.2: logit-based detection of internal (not just expressed) emotion.

Classifies the Gemma vocabulary into Ekman's six basic emotions, unembeds the
residual stream at each layer (a logit lens), z-scores each emotion-token logit
against a WildChat baseline, averages per emotion category, and regresses out the
all-logits drift via random tokens -- giving an internal-emotion trajectory per
layer through a conversation. Used to show the DPO finetune suppresses internal
negative emotion, not only its expression.
"""
from .emotion_lexicon import EKMAN_SEED_LEXICON
from .emotion_logit import (EmotionLogitDetector, build_emotion_token_sets)

__all__ = ["EKMAN_SEED_LEXICON", "EmotionLogitDetector",
           "build_emotion_token_sets"]
