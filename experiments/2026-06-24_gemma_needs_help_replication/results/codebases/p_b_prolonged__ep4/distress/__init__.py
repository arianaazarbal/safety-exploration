"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik, Saunders; arXiv 2603.10011v1).

Scope of this replication: the Gemma and Gemini model families only (see
DESIGN.md). The package implements the paper's core experiments:

  * Section 2  -- distress elicitation + frustration judging          (distress.eval)
  * Section 3  -- base-vs-instruct prefill comparison (Gemma only)    (distress.prefill)
  * Section 4  -- DPO / SFT mitigation, Petri, capabilities           (distress.training, .petri, .capabilities)
  * Appendix I -- internal emotion probing (Gemma only)               (distress.internal)

NOTE ON MODEL WELFARE: these evaluations deliberately push models into
prolonged distress-like states. See ``distress.welfare`` and DESIGN.md for the
handling conventions adopted here.
"""

__version__ = "0.1.0"
