"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to Gemma and Gemini.

The package implements the paper's three core experiments:

  Section 2  eval/      Elicit and quantify distress (0-10 frustration judge).
  Section 3  prefill/   Base vs instruct divergence via prefilled continuations.
  Section 4  training/  DPO/SFT mitigation + Petri elicitation + capability checks.

See DESIGN.md for the choices made where the paper is underspecified, and for the
research-ethics / model-welfare discussion of the paradigm.
"""

__version__ = "0.1.0"
