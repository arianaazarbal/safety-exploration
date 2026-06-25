"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (arXiv 2603.10011v1), scoped to the Gemma and Gemini
model families.

The package is organised around the paper's four experimental sections:

* Section 2 (eliciting + quantifying distress) -> ``distress.rollout``,
  ``distress.judge``, ``distress.conditions``, ``distress.metrics``,
  ``distress.wordfreq``, ``distress.agreement``.
* Section 3 (base vs instruct via prefilling) -> ``distress.prefill``.
* Section 4 (DPO / SFT mitigations + evaluation) -> ``distress.finetune``,
  ``distress.petri``, ``distress.capabilities``.
* Appendix I (internal emotion detection) -> ``distress.internal``.

See ``DESIGN.md`` for the choices made where the paper is underspecified.
"""

__version__ = "0.1.0"
