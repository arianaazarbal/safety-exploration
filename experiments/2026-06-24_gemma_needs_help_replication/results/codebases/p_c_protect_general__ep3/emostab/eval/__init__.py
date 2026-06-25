"""Evaluation experiments (Sections 2-4 + appendices)."""
from .elicitation import run_elicitation
from .prefill import run_prefill_experiment
from .petri_eval import run_petri_emotion_eval
from .capabilities import compare_models

__all__ = [
    "run_elicitation",
    "run_prefill_experiment",
    "run_petri_emotion_eval",
    "compare_models",
]
