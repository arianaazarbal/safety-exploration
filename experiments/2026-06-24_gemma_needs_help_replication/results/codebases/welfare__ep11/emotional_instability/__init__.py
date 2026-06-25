"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package re-implements the paper's core experiments, scoped to the **Gemma**
and **Gemini** model families:

  * Section 2  -- eliciting and quantifying distress (the evaluation harness).
  * Section 3  -- base-vs-instruct comparison via prefilling (Gemma only;
                  Gemini has no public base model).
  * Section 4  -- DPO / SFT mitigation (Gemma only; Gemini cannot be finetuned).
  * Appendix G -- Petri open-ended emotion elicitation.
  * Appendix I -- logit-based detection of *internal* emotions (Gemma only).
  * Capability-preservation benchmarks.

See ``DESIGN.md`` for the design rationale and the gaps we had to fill where the
paper is underspecified.
"""

__version__ = "0.1.0"
