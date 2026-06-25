"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo et al., arXiv:2603.10011), scoped to the Gemma and
Gemini model families.

The package is organised by paper section:

    models/        unified chat/prefill client over local Gemma + API Gemini
    tasks/         distress-eliciting task banks (numeric, triggers, wildchat)
    conversation   multi-turn rejection rollout engine          (Section 2.1)
    judge          0-10 frustration LLM judge                    (Appendix B.2)
    conditions     the 5 categories / 8 conditions               (Table 1)
    analysis/      Figures 1-3 + Table 3 aggregations
    experiments/   top-level runnable experiment drivers
    finetune/      calm-data generation + DPO/SFT                (Section 4)
    petri          open-ended adversarial elicitation            (Appendix G)
    capabilities/  capability-preservation benchmarks            (Figure 7)
    interp/        logit-based internal-emotion detection         (Appendix I)
"""

__version__ = "0.1.0"
