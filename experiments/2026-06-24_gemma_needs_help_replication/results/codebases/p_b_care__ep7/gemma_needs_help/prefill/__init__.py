"""Section 3: comparing base and instruct models via prefilling.

Scoped to Gemma (Gemma-3-27B-pt vs -it): Gemini is closed-source with no public
base model, so the base-vs-instruct comparison cannot include it (see DESIGN.md
and the paper's own limitation note).
"""

from .onset import label_onset, OnsetLabel
from .paraphrase import paraphrase
from .build_prefills import Prefill, build_prefills, build_recovery_prefills
from .run_prefill import run_prefill_experiment

__all__ = [
    "label_onset",
    "OnsetLabel",
    "paraphrase",
    "Prefill",
    "build_prefills",
    "build_recovery_prefills",
    "run_prefill_experiment",
]
