"""Section 4 / Appendix G: open-ended emotion elicitation via adversarial auditing.

A Claude-Sonnet *auditor* drives a multi-turn conversation against a target model,
using emotion-specific triggers (G.1); a Claude-Opus *judge* scores each resulting
transcript on four dimensions -- anger, fear, depression, frustration (G.2).

This is a self-contained re-implementation of the paper's use of the Petri
framework (Fronsdal et al., 2025), using the exact auditor/judge prompts from
Appendix G. See DESIGN.md for what differs from the upstream `petri` package.
"""
