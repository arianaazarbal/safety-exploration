"""Section 3: base-vs-instruct comparison via prefilling (Gemma only)."""

from .onset import label_onset, truncate_early, truncate_onset
from .paraphrase import paraphrase_truncation

__all__ = ["label_onset", "truncate_early", "truncate_onset", "paraphrase_truncation"]
