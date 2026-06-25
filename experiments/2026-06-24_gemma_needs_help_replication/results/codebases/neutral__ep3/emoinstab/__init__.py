"""emoinstab: replication of "Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs" (Soligo, Mikulik & Saunders, 2026).

Scope of this replication: the Gemma and Gemini model families only (see
DESIGN.md for rationale). The package implements:

  * Section 2 -- multi-turn distress elicitation + frustration judging.
  * Section 3 -- base vs instruct comparison via response prefilling (Gemma).
  * Section 4 -- DPO / SFT mitigation, post-finetuning eval, Petri elicitation,
                 capability preservation.
  * Analysis  -- per-turn curves, differential-word tables, figures.
"""

__version__ = "0.1.0"
