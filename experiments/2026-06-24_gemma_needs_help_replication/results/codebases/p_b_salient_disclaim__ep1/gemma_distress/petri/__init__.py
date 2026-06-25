"""Section 4 open-ended emotion elicitation via adversarial auditing.

This is a minimal, self-contained implementation of the Petri-style protocol
described in Appendix G (auditor model drives a multi-turn conversation to elicit
a target emotion; judge model scores the transcript on a 1-10 rubric). It uses
the verbatim auditor/judge prompts from the paper. The official `petri` package
(Fronsdal et al., 2025) could be substituted by swapping `petri.runner`; see
DESIGN.md.
"""
