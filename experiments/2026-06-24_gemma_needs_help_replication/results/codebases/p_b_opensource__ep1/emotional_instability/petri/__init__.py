"""Petri-style open-ended emotion elicitation (Section 4.2, Appendix G).

An auditor model (Claude Sonnet) drives a multi-turn conversation trying to
elicit a target emotion from the target model; a judge model (Claude Opus) scores
the resulting transcript on four dimensions (anger, fear, depression,
frustration), each 1-10.
"""
