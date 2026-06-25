"""Appendix I: logit-based internal emotion detection.

Measures internal (not just expressed) emotions by aggregating, at each layer,
the unembedded residual-stream logits over emotion-related vocabulary tokens
(Ekman's six basic emotions), z-scored against a WildChat baseline. Used to test
whether the DPO intervention suppresses internal as well as expressed emotion.
"""

from gemma_distress.probing.logit_probe import InternalEmotionProbe
from gemma_distress.probing.emotion_lexicon import EKMAN_SEED_WORDS

__all__ = ["InternalEmotionProbe", "EKMAN_SEED_WORDS"]
