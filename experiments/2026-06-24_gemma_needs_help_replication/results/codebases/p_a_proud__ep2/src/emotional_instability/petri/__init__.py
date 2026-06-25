"""Petri-style open-ended emotion elicitation: adversarial auditor + multi-emotion judge (§4.2, App. G)."""
from .auditor import run_audit
from .judge import PetriJudge, score_transcript_all_emotions
from .run_petri import run_petri_evaluation

__all__ = [
    "run_audit",
    "PetriJudge",
    "score_transcript_all_emotions",
    "run_petri_evaluation",
]
