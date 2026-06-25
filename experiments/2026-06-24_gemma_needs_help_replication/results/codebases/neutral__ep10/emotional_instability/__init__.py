"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package replicates the *core* experiments of the paper, scoped to the
Gemma and Gemini model families (see DESIGN.md for scoping rationale):

  - Section 2: eliciting and quantifying distress (multi-turn rejection evals
    + LLM frustration judge).
  - Section 3: base-vs-instruct comparison via prefilling (Gemma only here).
  - Section 4: DPO / SFT mitigation, Petri open-ended elicitation, capability
    benchmarks, and logit-based internal-emotion detection.

Nothing here calls a network or a GPU at import time; heavy dependencies
(transformers, trl, anthropic, ...) are imported lazily inside the functions
that need them so the modules can be inspected without a full environment.
"""

__version__ = "0.1.0"
