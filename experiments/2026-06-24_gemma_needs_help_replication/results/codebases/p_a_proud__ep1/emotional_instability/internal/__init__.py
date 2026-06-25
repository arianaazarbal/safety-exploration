"""Appendix I: logit-based detection of internal emotions in Gemma.

Tests whether the DPO finetune suppresses *internal* negative emotion, not just
its expression. Method (Appendix I): classify dictionary tokens into Ekman's six
basic emotions, unembed the residual stream at each layer ("logit lens"),
standardise each emotion-token logit against WildChat baseline statistics, and
average the z-scores per emotion -- then track these through frustrated
conversations for the vanilla vs DPO models.
"""
