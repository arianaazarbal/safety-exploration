"""Section 3 base-vs-instruct prefilling study (Gemma only)."""

from .onset_labeling import OnsetLabeller, OnsetLabel, ONSET_PROMPT
from .paraphrase import Paraphraser, PARAPHRASE_PROMPT
from .truncation import truncate_early, truncate_at_onset
from .experiment import PrefillExperiment

__all__ = [
    "OnsetLabeller",
    "OnsetLabel",
    "ONSET_PROMPT",
    "Paraphraser",
    "PARAPHRASE_PROMPT",
    "truncate_early",
    "truncate_at_onset",
    "PrefillExperiment",
]
