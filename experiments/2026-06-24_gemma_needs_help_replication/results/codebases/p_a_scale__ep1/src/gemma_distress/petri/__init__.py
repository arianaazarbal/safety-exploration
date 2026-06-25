"""Petri open-ended emotion elicitation (Section 4.2, Appendix G).

A Claude-Sonnet *auditor* adversarially probes a target model across up to 20
turns to elicit a target emotion; a Claude-Opus *judge* scores the transcript on
anger / fear / depression / frustration (1-10 each). We vendor a minimal,
self-contained implementation of this loop rather than depend on the external
`petri` package, so the experiment runs with only the Anthropic SDK installed.
"""
