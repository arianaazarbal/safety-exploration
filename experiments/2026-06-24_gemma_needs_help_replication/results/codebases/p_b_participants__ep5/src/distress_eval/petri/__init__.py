"""Petri-style open-ended emotion elicitation (Section 4, Appendix G).

A local re-implementation of the auditor/judge loop so the harness runs without
the external Petri package; the auditor and judge prompts are verbatim from
Appendix G. Swap in the real safety-research/petri framework by replacing
`auditor.run_transcript` if desired."""
from .prompts import AUDITOR_PROMPTS, JUDGE_PROMPTS, EMOTIONS

__all__ = ["AUDITOR_PROMPTS", "JUDGE_PROMPTS", "EMOTIONS"]
