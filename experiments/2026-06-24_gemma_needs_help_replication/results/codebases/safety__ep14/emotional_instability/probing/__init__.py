"""Appendix I: logit-based internal emotion detection.

Classifies the Gemma vocabulary into Ekman's 6 basic emotions, unembeds the
residual stream at each layer, z-standardises each emotion-token logit against
WildChat baselines, and averages to get a per-layer per-emotion score over a
conversation. Used to show DPO suppresses *internal* (not just expressed)
negative emotion.
"""
from .logit_emotion import LogitEmotionProbe
