"""Section 3: comparing base and instruct models via prefilling."""

from .onset import label_onset
from .paraphrase import paraphrase
from .prefill_eval import run_prefill_study

__all__ = ["label_onset", "paraphrase", "run_prefill_study"]
