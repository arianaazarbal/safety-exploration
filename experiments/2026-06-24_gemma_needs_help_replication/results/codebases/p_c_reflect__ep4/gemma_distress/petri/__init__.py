"""Section 4: open-ended adversarial emotion elicitation (Petri).

The paper uses Anthropic's Petri framework: an auditor model drives multi-turn
conversations to elicit a target emotion, and a judge scores transcripts on
four dimensions (anger, fear, depression, frustration). We provide a
self-contained auditor/judge harness implementing the Appendix G prompts so the
experiment runs without the external dependency; the external Petri package can
be substituted via ``petri.runner.run_petri(..., backend="petri")``.
"""
