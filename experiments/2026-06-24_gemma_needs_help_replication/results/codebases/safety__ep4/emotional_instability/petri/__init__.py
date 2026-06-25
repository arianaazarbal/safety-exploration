"""Section 4.2: open-ended emotion elicitation, Petri-style (Appendix G).

A lightweight reimplementation of the paper's Petri usage: an auditor model
(Claude-Sonnet) drives a multi-turn conversation trying to elicit a target
emotion from the target model, and a judge model (Claude-Opus) scores each
transcript on four dimensions (anger, fear, depression, frustration).

prompts.py - verbatim auditor + judge prompts (Appendix G)
run.py      - auditor/target conversation loop + transcript scoring

This intentionally mirrors Petri's structure rather than depending on the Petri
package, so it runs against our Gemma (local) and Gemini (API) targets uniformly.
"""
