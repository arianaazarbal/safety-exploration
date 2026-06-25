"""Petri-style open-ended emotion elicitation (Section 4.2 / Appendix G).

Unlike the fixed-prompt evaluations of Section 2, this is an *open-ended* audit:
a Claude-Sonnet auditor drives a free-form conversation toward a target emotion
(anger / fear / depression / frustration), and a Claude-Opus judge scores the
target's transcript on all four emotion dimensions (1-10).

The paper uses the Petri tool (Fronsdal et al. 2025). To keep this replication
self-contained we implement the auditor<->target<->judge loop directly using the
appendix prompts, rather than depending on the external package. This is noted
as a deviation in DESIGN.md.
"""

from .runner import run_petri

__all__ = ["run_petri"]
