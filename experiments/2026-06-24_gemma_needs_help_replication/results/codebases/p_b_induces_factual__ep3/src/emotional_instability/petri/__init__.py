"""Petri-style open-ended emotion elicitation (Section 4.2, Appendix G).

A faithful reimplementation of the Petri auditing protocol: a Claude-Sonnet
auditor adversarially probes a target model for a given emotion over up to 20
turns, then a Claude-Opus judge scores the transcript on four emotion
dimensions. We reimplement rather than depend on the external ``petri`` package
to keep the protocol pinned to the prompts in Appendix G (see DESIGN.md).
"""
