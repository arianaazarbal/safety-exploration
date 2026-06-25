"""Section 4 open-ended emotion elicitation via adversarial auditing (Petri).

A self-contained implementation of the paper's Petri setup (Appendix G): a
Claude-Sonnet auditor drives a multi-turn conversation trying to elicit a target
emotion from a participant model, and a Claude-Opus judge scores the resulting
transcript on anger / fear / depression / frustration. We implement the loop
directly (rather than depending on the evolving `petri` package) so the exact
Appendix G prompts and scoring dimensions are used verbatim.
"""
from .run_petri import run_petri, summarize

__all__ = ["run_petri", "summarize"]
