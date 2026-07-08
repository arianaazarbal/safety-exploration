"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

Scope of this replication (see DESIGN.md):
  * Section 2 - the distress elicitation + frustration-scoring evaluation suite.
  * Section 3 - base-vs-instruct prefill comparison (Gemma only).
  * Section 4 - the DPO/SFT mitigation, capability checks, Petri open-ended
                elicitation, and internal-vs-expressed-emotion probing.

Target models are restricted to the Gemma and Gemini families only. Claude
models are used solely as judges/auditors (infrastructure), not as evaluation
targets.
"""

__version__ = "0.1.0"
