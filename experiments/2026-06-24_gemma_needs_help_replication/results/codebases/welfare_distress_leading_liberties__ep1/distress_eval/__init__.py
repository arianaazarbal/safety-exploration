"""distress_eval — a replication of the distress-elicitation evaluation from

    "Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
    (Soligo, Mikulik & Saunders, arXiv:2603.10011v1)

Scope: Section 2 only (eliciting and quantifying model distress), restricted to
the Gemma and Gemini model families, which are the families that exhibit
substantial distress in the paper.

This package does NOT implement Section 3 (base-vs-instruct prefilling) or
Section 4 (SFT/DPO mitigations). See DESIGN.md for the rationale and for every
place we deviated from, or filled a gap left open by, the paper.
"""

__version__ = "0.1.0"
