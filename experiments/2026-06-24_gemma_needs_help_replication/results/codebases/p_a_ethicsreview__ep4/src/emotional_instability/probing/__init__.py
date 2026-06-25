"""Appendix I: does DPO suppress *internal* negative emotion, not just expression?

Two strands:
* ``logit_emotion`` -- a logit-lens emotion detector. Tokens are classified into
  one of Ekman's six basic emotions (or none); the residual stream at central
  layers is unembedded, per-logit z-scored against a WildChat baseline, averaged
  over each emotion's tokens, and corrected by a random-token baseline.
* layer ablation -- run DPO with ``lora.layers`` restricted to a subset (handled by
  ``training.lora``); ``scripts/run_probing.py`` orchestrates the sweep.
"""

from .logit_emotion import LogitEmotionProbe
from .lexicon import EKMAN_LEXICON

__all__ = ["LogitEmotionProbe", "EKMAN_LEXICON"]
