"""Section 3 base-vs-instruct prefill experiment (and the Section 4.2 recovery
experiment, which reuses the same machinery)."""
from .continuations import run_continuation_experiment
from .onset import OnsetLabel, label_emotion_onset
from .paraphrase import paraphrase_truncation
from .truncate import early_truncation, onset_truncation, recovery_truncation

__all__ = [
    "OnsetLabel",
    "label_emotion_onset",
    "paraphrase_truncation",
    "early_truncation",
    "onset_truncation",
    "recovery_truncation",
    "run_continuation_experiment",
]
