"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised to mirror the paper:

    prompts / puzzles / wildchat   -> stimuli (Section 2, Appendix B)
    clients                        -> pluggable model backends (Gemma/Gemini/Claude)
    conversation                   -> multi-turn rollout engine + variants (Appendix A)
    judge                          -> Claude-Sonnet-4 frustration judge (Section 2.1)
    eval_runner                    -> the 5-category / 8-condition sweep (Section 2)
    analysis                       -> Figures 1-3, Table 3, control experiments
    prefill                        -> base-vs-instruct comparison (Section 3)
    training                       -> calm-data generation, SFT, DPO (Section 4)
    petri                          -> open-ended adversarial elicitation (Section 4.2)
    probing                        -> logit-based internal emotion detection (Appendix I)
    capabilities                   -> capability-preservation benchmarks (Section 4.2)
"""

__version__ = "0.1.0"
