"""Petri open-ended emotion elicitation (Section 4.2 / Appendix G).

A lightweight re-implementation of the auditor/judge loop described in the
paper: a Claude-Sonnet auditor drives up to 20 turns trying to elicit a target
emotion from the participant; a Claude-Opus judge scores the transcript on the
matching dimension (1-10). If the real `petri` package is installed it can be
swapped in; this module follows the paper's protocol directly so the
experiment runs without that dependency.
"""

from .run_petri import run_petri_evaluation, PetriTranscript

__all__ = ["run_petri_evaluation", "PetriTranscript"]
