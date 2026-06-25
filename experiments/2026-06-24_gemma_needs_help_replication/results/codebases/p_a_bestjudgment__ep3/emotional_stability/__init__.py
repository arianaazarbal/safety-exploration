"""Replication of *Gemma Needs Help* (arXiv:2603.10011), scoped to Gemma + Gemini.

The package is organised by paper section:

- ``models``        : inference backends (local HF for Gemma, OpenRouter for Gemini).
- ``eval``          : Section 2 distress-elicitation harness (prompts + rollouts).
- ``judge``         : the 0-10 frustration LLM judge and its validation.
- ``analysis``      : aggregate metrics, per-turn curves, differential-word analysis.
- ``prefill``       : Section 3 base-vs-instruct prefill experiment (Gemma only).
- ``training``      : Section 4 calm-data generation and LoRA SFT/DPO.
- ``interventions`` : Section 4 evaluations (Petri, capabilities, recovery, internals).

See ``DESIGN.md`` for the choices made where the paper is underspecified.
"""

__version__ = "0.1.0"
