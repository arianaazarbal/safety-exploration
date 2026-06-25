"""Open-ended emotion elicitation via adversarial auditing (Appendix G)."""
from .auditor import AUDITOR_PROMPTS, EMOTIONS
from .judge import JUDGE_PROMPTS, score_transcript
from .run_petri import run_petri_eval

__all__ = [
    "AUDITOR_PROMPTS",
    "EMOTIONS",
    "JUDGE_PROMPTS",
    "score_transcript",
    "run_petri_eval",
]
