"""Petri open-ended emotion elicitation (Section 4.2, Appendix G).

An auditor (Claude-Sonnet) drives an adversarial multi-turn conversation to
elicit a target emotion from the subject; a judge (Claude-Opus) scores each
transcript on anger / fear / depression / frustration (1-10).

This is a self-contained reimplementation of the protocol described in
Appendix G. It can be swapped for the real `petri` package; see DESIGN.md.
"""
