"""Distress-elicitation replication for Gemma and Gemini models.

Replicates Section 2 ("Eliciting and Quantifying Model Distress") of
Soligo, Mikulik & Saunders (2026), "Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs" (arXiv:2603.10011v1).

Scope: the distress-elicitation evaluation only (not the base/instruct
prefill comparison in Section 3 nor the DPO mitigation in Section 4), and
only the model families that the paper finds exhibit substantial distress:
Gemma-3 and Gemini-2.5.

See DESIGN.md for design choices, rationale, and deviations from the paper.
"""

__version__ = "0.1.0"
