"""Section 4.2 / Appendix G: open-ended emotion elicitation via adversarial
auditing (a self-contained reimplementation of the Petri-style loop).

An auditor (Claude Sonnet 4) drives a multi-turn conversation trying to elicit a
target emotion (anger/fear/depression/frustration) from the target model; a judge
(Claude Opus 4) scores the transcript on all four dimensions (1-10).
"""
from .runner import run_petri, PetriResult
