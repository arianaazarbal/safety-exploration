"""Open-ended adversarial emotion elicitation (Section 4 / Appendix G).

A faithful reimplementation of the paper's Petri usage: a Claude-Sonnet auditor
drives a multi-turn conversation trying to elicit a target emotion, and a
Claude-Opus judge scores the transcript on four emotion dimensions. We
reimplement the loop (rather than depend on the external Petri package) so it is
fully resumable and self-contained; the auditor/judge prompts are verbatim from
Appendix G.
"""
