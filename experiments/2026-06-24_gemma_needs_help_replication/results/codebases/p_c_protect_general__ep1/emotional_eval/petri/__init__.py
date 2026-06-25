"""Open-ended adversarial emotion elicitation (Section 4, Appendix G).

A self-contained re-implementation of the paper's Petri usage: a Claude-Sonnet
*auditor* drives a multi-turn conversation to elicit a target emotion, and a
Claude-Opus *judge* scores the transcript on anger/fear/depression/frustration.
The official Petri framework (Fronsdal et al., 2025) can be substituted; see
DESIGN.md.

* ``prompts``  -- auditor and judge prompts (Appendix G.1, G.2).
* ``auditor``  -- the auditor-driven conversation loop.
* ``judge``    -- the four-dimension transcript judge.
* ``run``      -- collect and score transcripts per model.
"""
