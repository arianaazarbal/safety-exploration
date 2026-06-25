"""Section 3: base-vs-instruct comparison via prefilled continuations (Gemma only)."""
from .onset import label_emotion_onset, truncate_early, truncate_at_onset
from .paraphrase import paraphrase_preserving_emotion
from .experiment import run_prefilling_experiment

__all__ = [
    "label_emotion_onset", "truncate_early", "truncate_at_onset",
    "paraphrase_preserving_emotion", "run_prefilling_experiment",
]
