"""Section 4.1 — Petri-style open-ended emotion elicitation (Appendix G).

A self-contained reimplementation of the Petri auditor/judge pattern: a Claude
auditor drives a multi-turn conversation against the target model trying to
elicit a target emotion, and a Claude-Opus judge scores the transcript on four
dimensions (anger, fear, depression, frustration). This is not the upstream
``petri`` package; see DESIGN.md for why and how to swap it in.
"""

from .run_petri import PetriRunner, PetriResult  # noqa: F401
