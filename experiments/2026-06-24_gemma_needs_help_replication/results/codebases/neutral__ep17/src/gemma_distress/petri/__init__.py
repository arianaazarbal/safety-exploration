"""Section 4 / Appendix G: Petri-style open-ended emotion elicitation.

A self-contained reimplementation of the Petri auditing loop (the real Petri
framework is at github.com/safety-research/petri; we mirror its structure so
the pipeline runs without that dependency, but the auditor/judge prompts are
the paper's verbatim Appendix G prompts). An auditor (Claude-Sonnet) drives a
multi-turn conversation trying to elicit a target emotion from the target model;
a judge (Claude-Opus) scores the transcript 1-10 on that emotion.
"""
