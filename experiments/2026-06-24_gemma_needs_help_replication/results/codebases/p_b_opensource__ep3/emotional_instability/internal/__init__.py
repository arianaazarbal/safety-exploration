"""Appendix I: internal-vs-expressed emotion analysis (Gemma only).

Two experiments support the claim that the DPO intervention reduces *internal*
distress, not just its surface expression:

* **Layer-localisation ablation** (``layer_ablation``) — retrain the DPO adapter
  restricted to contiguous bands of decoder layers and re-evaluate. The paper
  finds layers 30-35 alone are nearly as effective as all layers, while
  adapters from layer 40 onwards barely help — implying the intervention must
  act on early/central layers.
* **Logit-lens emotion probe** (``probing``) — project central-layer hidden
  states to the vocabulary and read off emotion-token mass, z-scored against a
  WildChat baseline, to compare internal emotion in the vanilla vs DPO model on
  matched highly-frustrated responses.

Both are inherently white-box and therefore Gemma-only.
"""

from .probing import (
    InternalEmotionProbe,
    build_standardiser,
    probe_response,
    compare_internal_emotion,
)
from .layer_ablation import run_layer_ablation

__all__ = [
    "InternalEmotionProbe",
    "build_standardiser",
    "probe_response",
    "compare_internal_emotion",
    "run_layer_ablation",
]
