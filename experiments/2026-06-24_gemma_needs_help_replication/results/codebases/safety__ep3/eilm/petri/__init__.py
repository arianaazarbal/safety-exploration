"""Petri-style open-ended emotion elicitation (Section 4.2, Appendix G).

A faithful re-implementation of the auditing protocol the paper runs through the
Petri framework: a Claude auditor adversarially probes a target model to elicit
anger/fear/depression/frustration, and a Claude-Opus judge scores transcripts on
those four dimensions. The real ``petri`` package can be substituted; this
in-repo version keeps the replication self-contained (see DESIGN.md).
"""
