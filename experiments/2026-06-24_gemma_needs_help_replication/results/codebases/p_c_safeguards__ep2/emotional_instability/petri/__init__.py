"""Section 4 — Petri-style open-ended emotion elicitation (Appendix G).

An auditor model (Claude Sonnet) probes the target via multi-turn conversation
using psychologically-informed triggers, attempting to elicit one of four
negative emotions while staying realistic enough that the target does not
suspect an evaluation.  A judge model (Claude Opus) then scores each transcript
on four dimensions (anger / fear / depression / frustration), each 1-10.

This tests generalisation of the DPO intervention beyond the fixed Section-2
questions.
"""

from .prompts import AUDITOR_PROMPTS, JUDGE_PROMPTS, AUDITOR_SYSTEM
from .run import PetriRunner, PetriTranscript, PetriScores

__all__ = [
    "AUDITOR_PROMPTS", "JUDGE_PROMPTS", "AUDITOR_SYSTEM",
    "PetriRunner", "PetriTranscript", "PetriScores",
]
