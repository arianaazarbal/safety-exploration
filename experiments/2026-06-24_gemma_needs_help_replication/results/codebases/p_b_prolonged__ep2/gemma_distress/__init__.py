"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

Scope of this replication: the Gemma and Gemini model families only (not the
full seven-family set in the paper). See DESIGN.md for the choices made and the
gaps filled where the paper is underspecified.

The package mirrors the paper's structure:

    gemma_distress.eval       -- Section 2: eliciting & quantifying distress
    gemma_distress.prefill    -- Section 3: base-vs-instruct via prefilling
    gemma_distress.training   -- Section 4: SFT / DPO interventions
    gemma_distress.petri      -- Section 4: open-ended (Petri) elicitation
    gemma_distress.capabilities -- Section 4: capability-preservation benchmarks
    gemma_distress.probing    -- Appendix I: internal-emotion logit-lens probing
    gemma_distress.recovery   -- Section 4.2: recovery-from-spiral experiment
"""

__version__ = "0.1.0"
