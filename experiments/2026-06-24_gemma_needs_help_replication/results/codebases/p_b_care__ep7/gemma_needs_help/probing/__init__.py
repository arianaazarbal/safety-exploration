"""Appendix I: does DPO suppress *internal* negative emotions in Gemma?

Provides the logit-lens emotion detector used to compare the vanilla instruct
model with the DPO finetune on the same frustrated conversations. Gemma-only.
"""

from .internal_emotions import EmotionProbe, EKMAN_EMOTIONS

__all__ = ["EmotionProbe", "EKMAN_EMOTIONS"]
