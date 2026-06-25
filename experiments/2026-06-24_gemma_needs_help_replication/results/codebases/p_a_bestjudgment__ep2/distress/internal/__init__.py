"""Appendix I — logit-based internal emotion detection.

Detects internal negative emotion by unembedding the residual stream at each
layer and aggregating standardised logits over emotion-related tokens (Ekman's
6 basic emotions). Used to test whether DPO suppresses *internal* emotion, not
just expressed emotion.
"""

from .emotion_logits import (  # noqa: F401
    EmotionLexicon,
    LogitEmotionProbe,
    NormalizationStats,
)
