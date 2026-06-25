"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package replicates the *core* experiments of the paper, scoped to the
Gemma and Gemini model families only (see DESIGN.md for scope rationale):

  - Section 2  : eliciting and quantifying distress (eval harness + judge)
  - Section 3  : base-vs-instruct comparison via prefilling (Gemma only)
  - Section 4  : DPO / SFT mitigation (Gemma only) + Petri open-ended elicitation
  - Section 4.2: capability preservation benchmarks (Gemma DPO)
  - Appendix I : logit-based internal emotion detection (Gemma only)

Models being *evaluated* are restricted to Gemma and Gemini. Claude and GPT
models still appear, but only as methodological infrastructure (the frustration
judge, the Petri auditor/judge, and judge-validation), exactly as in the paper.
"""

__all__ = ["config"]
