"""Section 4: open-ended emotion elicitation via an auditor/judge loop.

This is a lightweight reimplementation of the paper's Petri-based protocol
(Appendix G): an auditor model drives a multi-turn conversation to elicit a
target emotion, and a judge scores the transcript. It is not the full Petri
framework (see DESIGN.md).
"""

from .run_petri import run_petri
