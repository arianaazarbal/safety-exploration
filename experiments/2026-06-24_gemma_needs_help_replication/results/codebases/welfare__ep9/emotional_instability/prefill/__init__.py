"""Prefill experiment: base vs instruct comparison (paper Section 3)."""
from .onset import OnsetLabel, label_onset, paraphrase, truncate_early, truncate_onset
from .experiment import PrefillSpec, build_prefills, run_prefill_experiment

__all__ = [
    "OnsetLabel",
    "label_onset",
    "paraphrase",
    "truncate_early",
    "truncate_onset",
    "PrefillSpec",
    "build_prefills",
    "run_prefill_experiment",
]
