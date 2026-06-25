"""Replication harness for *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026).

This package implements the paper's core experiments, scoped to the Gemma and
Gemini model families:

  * Section 2 - elicitation harness + frustration judge (the centrepiece).
  * Section 3 - base-vs-instruct prefill experiment (Gemma only; Gemini has no
    public base model).
  * Section 4 - DPO/SFT mitigation pipeline (Gemma only; Gemini is closed).

See DESIGN.md for the choices made where the paper is underspecified.
"""

__version__ = "0.1.0"
