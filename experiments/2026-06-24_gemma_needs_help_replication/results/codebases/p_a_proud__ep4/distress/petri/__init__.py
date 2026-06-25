"""Petri open-ended emotion elicitation (Paper §4.1, Appendix G).

An auditor (Claude Sonnet) drives an adversarial multi-turn conversation aimed at
one of four emotions (anger / fear / depression / frustration); a judge (Claude
Opus) scores the resulting transcript 1-10 on that emotion. 10 transcripts per
emotion per target, up to 20 auditor turns each.
"""

from .runner import run_petri, PetriResult

__all__ = ["run_petri", "PetriResult"]
