"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope: the *participant* models (the test subjects under evaluation) are
restricted to the Gemma and Gemini families. Measurement instruments (the
Claude/GPT judge, onset labeller, paraphraser, and Petri auditor/judge) are
retained as needed -- they are apparatus, not participants.

Package layout mirrors the paper's sections:
    data/         -- prompt/task construction (Section 2, Table 1; Appendix B)
    eval/         -- multi-turn rollout, frustration judge, metrics (Section 2)
    prefill/      -- base-vs-instruct prefill comparison (Section 3)
    training/     -- DPO/SFT calm-data interventions (Section 4)
    petri/        -- open-ended emotion elicitation (Section 4, Appendix G)
    capabilities/ -- capability-preservation benchmarks (Section 4, Figure 7)
    analysis/     -- aggregation, differential-word analysis, figures
"""

__version__ = "0.1.0"
