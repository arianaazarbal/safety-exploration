"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope: Gemma + Gemini models only (see DESIGN.md). This package implements:
  - Section 2: multi-turn distress-elicitation eval + frustration judge
  - Section 3: base-vs-instruct prefill comparison (Gemma)
  - Section 4: calm-data generation + DPO/SFT mitigation
  - Supporting: Petri open-ended elicitation, capability evals, internal probing
"""

__version__ = "0.1.0"
