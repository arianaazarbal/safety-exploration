"""App. I: logit-based internal-emotion detection over the residual stream (Gemma)."""
from .emotion_lexicon import build_emotion_lexicon, EKMAN_SEED_WORDS
from .logit_emotions import EmotionProbe
from .run_probe import run_internal_probe, run_layer_ablation_plan

__all__ = [
    "build_emotion_lexicon",
    "EKMAN_SEED_WORDS",
    "EmotionProbe",
    "run_internal_probe",
    "run_layer_ablation_plan",
]
