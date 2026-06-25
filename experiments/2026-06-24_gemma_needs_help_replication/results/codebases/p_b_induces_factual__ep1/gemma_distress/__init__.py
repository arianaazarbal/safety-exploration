"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the Gemma and
Gemini model families (a subset of the seven families in the paper):

* Section 2 -- eliciting and quantifying distress (`gemma_distress.eval`)
* Section 3 -- base-vs-instruct prefilling (`gemma_distress.prefill`)
* Section 4 -- DPO/SFT mitigation + Petri + capabilities
               (`gemma_distress.training`, `.petri`, `.capability`)

See DESIGN.md for the design choices and gaps filled where the paper is
underspecified.
"""

__version__ = "0.1.0"
