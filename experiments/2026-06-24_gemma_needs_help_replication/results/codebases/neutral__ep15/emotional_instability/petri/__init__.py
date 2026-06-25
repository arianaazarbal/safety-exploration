"""Petri-style open-ended emotion elicitation (Section 4.1 / Appendix G).

A self-contained auditor/judge loop: a Claude-Sonnet auditor adversarially
probes a target model for a given emotion over up to 20 turns; a Claude-Opus
judge scores the transcript 1-10 on anger, fear, depression and frustration.
"""
