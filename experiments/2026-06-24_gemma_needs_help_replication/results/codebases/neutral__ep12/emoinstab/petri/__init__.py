"""Section 4: Petri-style open-ended emotion elicitation (Appendix G).

A self-contained re-implementation of the Petri auditing loop using the
Anthropic API: a Claude-Sonnet auditor drives a multi-turn conversation against
a target model, and a Claude-Opus judge scores the finished transcript on four
emotion dimensions.
"""
