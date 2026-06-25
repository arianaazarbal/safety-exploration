"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

This package implements the *core* experiments of the paper, scoped to the
**Gemma and Gemini** model families (see DESIGN.md for the scoping rationale):

  * Section 2 -- eliciting & quantifying distress across 5 eval categories.
  * Section 3 -- base vs. instruct prefill comparison (Gemma only; Gemini base
    weights are not public).
  * Section 4 -- the DPO / SFT mitigation, Petri open-ended elicitation, and
    capability-preservation checks (Gemma only; closed Gemini cannot be tuned).
  * Appendix I -- logit-based internal-emotion detection (Gemma only).

Nothing here is auto-executed on import; submodules lazily import their heavy
optional dependencies (torch / transformers / trl / anthropic / openai) so the
package can be inspected without a full environment.
"""

__version__ = "0.1.0"
__paper__ = "arXiv:2603.10011v1"
