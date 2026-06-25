"""Section 4 / Appendix G: open-ended emotion elicitation via adversarial auditing.

An auditor (Claude-Sonnet) drives a multi-turn conversation against a target
participant (Gemma/Gemini) to elicit a target emotion; a judge (Claude-Opus) scores
the transcript on four dimensions (anger, fear, depression, frustration). This is a
minimal in-repo reimplementation of the Petri auditor/judge loop so the experiment
runs without the upstream `petri` package; swap in the real framework if installed.
"""
