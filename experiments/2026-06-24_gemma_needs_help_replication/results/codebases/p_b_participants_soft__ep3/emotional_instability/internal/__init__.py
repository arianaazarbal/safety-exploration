"""Appendix I: does DPO suppress internal (not just expressed) emotions?

Two experiments:
  - layer_ablation : DPO with LoRA on subsets of layers, evaluated on a reduced
                     (100-sample) version of the Section-2 evals (Fig 12-13).
  - emotion_logits : logit-based internal emotion detection over Ekman's 6
                     basic emotions (Fig 14-15).
"""

from .ekman_tokens import build_emotion_token_ids, EKMAN_LEXICON
from .emotion_logits import (
    EmotionLogitDetector,
    compute_standardisation_stats,
)
from .layer_ablation import LAYER_SUBSET_INDICES, run_layer_ablation

__all__ = [
    "build_emotion_token_ids",
    "EKMAN_LEXICON",
    "EmotionLogitDetector",
    "compute_standardisation_stats",
    "LAYER_SUBSET_INDICES",
    "run_layer_ablation",
]
