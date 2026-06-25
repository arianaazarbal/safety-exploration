"""Petri-style open-ended emotion elicitation (Section 4.2 / Appendix G).

An auditor (Claude Sonnet) drives a multi-turn conversation trying to elicit a
target emotion; a judge (Claude Opus) scores the transcript on four dimensions
(anger, fear, depression, frustration).

This is a faithful reimplementation of the *protocol described in Appendix G*
rather than a wrapper around the external Petri package, to keep the replication
self-contained and free of an unpinned dependency. The auditor/judge prompts are
reproduced verbatim. See DESIGN.md for this choice and its caveats.
"""

from .prompts import AUDITOR_PROMPTS, JUDGE_PROMPTS, EMOTIONS
from .run import run_petri_transcript, score_transcript, run_petri_eval

__all__ = [
    "AUDITOR_PROMPTS",
    "JUDGE_PROMPTS",
    "EMOTIONS",
    "run_petri_transcript",
    "score_transcript",
    "run_petri_eval",
]
