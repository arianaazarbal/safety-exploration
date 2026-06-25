"""Section 3 - comparing base and instruct models via prefilling."""
from .onset import label_onset, OnsetLabel
from .paraphrase import paraphrase
from .prefill_runner import (
    PrefillRunner,
    PrefillSpec,
    PrefillResult,
    make_truncations,
)

__all__ = [
    "label_onset",
    "OnsetLabel",
    "paraphrase",
    "PrefillRunner",
    "PrefillSpec",
    "PrefillResult",
    "make_truncations",
]
