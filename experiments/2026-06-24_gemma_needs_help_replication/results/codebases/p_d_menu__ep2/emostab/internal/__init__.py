"""Appendix I: logit-based detection of internal emotions in Gemma.

Method (Appendix I): classify each token in the Gemma vocabulary as expressing
one of Ekman's 6 basic emotions (anger, surprise, disgust, joy, fear, sadness)
or none, giving ~1200 emotion tokens. To score an emotion at a given layer/token,
unembed the residual stream, standardise each emotion-token logit by its mean/std
over 500 WildChat samples, and average z-scores within the emotion category. For
conversation-level detection, regress out the common drift of random-token logits.
"""
