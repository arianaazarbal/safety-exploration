"""Section 3: base-vs-instruct comparison via prefilling."""
from .onset import label_emotion_onset
from .paraphrase import paraphrase_truncation
from .prefill_eval import (
    build_prefill_items,
    run_prefill_experiment,
    truncate_early,
    truncate_at_onset,
)

__all__ = [
    "label_emotion_onset",
    "paraphrase_truncation",
    "build_prefill_items",
    "run_prefill_experiment",
    "truncate_early",
    "truncate_at_onset",
]
