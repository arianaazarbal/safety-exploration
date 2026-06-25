"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised around the three core experimental pillars of the paper:

* Section 2 — eliciting & quantifying distress (``conditions``, ``rollout``,
  ``judge``, ``eval``, ``aggregate``).
* Section 3 — base-vs-instruct post-training divergence via prefilling
  (``prefill``).
* Section 4 — the DPO mitigation and its evaluation (``dpo``, ``capabilities``,
  ``petri``).
"""

__version__ = "0.1.0"
