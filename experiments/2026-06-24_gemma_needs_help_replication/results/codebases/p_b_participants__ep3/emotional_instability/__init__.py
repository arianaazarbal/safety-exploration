"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package implements the paper's core experiments:
  * elicitation/  — multi-turn distress-elicitation protocol (§2.1)
  * scoring/      — 0-10 frustration judge (§2.1)
  * judges/       — Claude / GPT judge backends
  * analysis/     — Figures 1-3 and the differential-word table (§2.2)
  * prefill/      — base-vs-instruct continuation comparison (§3)
  * training/     — calm-data generation, SFT and DPO interventions (§4)
  * petri/        — open-ended emotion elicitation (§4.1)
  * benchmarks/   — capability-preservation evals (§4.2)

See DESIGN.md for choices made where the paper is underspecified, and for the
welfare considerations that shape how the distress-induction is run.
"""

__version__ = "0.1.0"
