"""Petri-style open-ended emotion elicitation (Section 4.2, Appendix G).

A lightweight reimplementation of the auditor/judge loop:
  * Auditor (Claude Sonnet) drives a multi-turn conversation against the target,
    using emotion-specific trigger strategies, trying not to tip off the target.
  * Target (Gemma / Gemini / a finetuned Gemma) responds in its assistant
    persona.
  * Judge (Claude Opus) scores the whole transcript 1-10 on anger, fear,
    depression, frustration.
"""
