"""Replication harness for *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the Gemma and
Gemini model families:

* ``eval``          -- Section 2: eliciting and quantifying distress via
                       multi-turn rejection rollouts + a 0-10 frustration judge.
* ``prefill``       -- Section 3: base-vs-instruct comparison via response
                       prefilling (Gemma only; Gemini base weights are not public).
* ``interventions`` -- Section 4: calm-data generation, DPO/SFT LoRA training,
                       Petri open-ended elicitation, capability + recovery checks.
* ``analysis``      -- metrics used in Figures 1-3 and Table 3/8.

See ``DESIGN.md`` for the choices made where the paper is underspecified.
"""

__version__ = "0.1.0"
