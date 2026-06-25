"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026, arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the Gemma and
Gemini model families (see DESIGN.md for the scoping rationale):

  * eval/        Section 2 - eliciting and quantifying distress across models.
  * prefill/     Section 3 - base-vs-instruct comparison via prefilling (Gemma).
  * training/    Section 4 - SFT and DPO mitigation finetunes of Gemma-3-27B-it.
  * capabilities/ Section 4 - capability-preservation benchmarks.
  * petri/       Section 4 - open-ended emotion elicitation via Petri.
  * analysis/    Figures and tables.

Nothing here is model-family specific except the local-weights paths; the
backends abstraction (``backends/``) hides whether a model is served locally
(Gemma) or over an API (Gemini, Claude judge).
"""

__version__ = "0.1.0"
