"""Petri-style open-ended emotion elicitation (Section 4 / Appendix G).

A self-contained reimplementation of the auditor/judge loop using the exact
Appendix G prompts (auditor = Claude-Sonnet, judge = Claude-Opus). We reimplement
rather than depend on the external ``petri`` package so the harness is
self-contained and version-stable; the structure mirrors Petri's design. See
DESIGN.md.
"""
