"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik, Saunders; arXiv 2603.10011v1).

Scope: Gemma and Gemini model families only (see DESIGN.md).

This package implements the paper's core experiments:
  - emoeval.eval      : Section 2  — eliciting & quantifying distress
  - emoeval.prefill   : Section 3  — base vs instruct via prefilling (Gemma)
  - emoeval.training  : Section 4  — DPO / SFT mitigation (Gemma)
  - emoeval.petri     : Section 4.2 — open-ended elicitation
  - emoeval.probing   : Appendix I — internal (logit-based) emotion detection
  - emoeval.capabilities : Section 4.2 — capability-preservation benchmarks

See emoeval.welfare for the model-welfare guardrails that wrap distress
elicitation.
"""

__version__ = "0.1.0"
