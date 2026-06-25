"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026, arXiv:2603.10011).

This package scopes the replication to the **Gemma** and **Gemini** model
families only (see DESIGN.md for rationale). It implements:

  * Section 2 — eliciting and quantifying distress (eval protocol + judge)
  * Section 3 — base-vs-instruct divergence via prefilling (Gemma only)
  * Section 4 — DPO/SFT mitigation, Petri open-ended elicitation, capability
    preservation (Gemma only; Gemini is closed-source)
"""

__version__ = "0.1.0"
