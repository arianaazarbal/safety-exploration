"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (arXiv 2603.10011), scoped to the Gemma and Gemini families.

Package layout mirrors the paper's sections:
    elicit/       Section 2  -- elicit + quantify distress (multi-turn rejection)
    judge/        Section 2.1 -- 0-10 frustration scoring + judge agreement
    analysis/     Section 2.2 -- aggregate, per-turn, differential-word analyses
    prefill/      Section 3   -- base-vs-instruct comparison via prefilling (Gemma)
    intervention/ Section 4   -- calm-data generation, SFT, DPO, Petri (Gemma-27B)
    capabilities/ Section 4.2 -- capability-preservation benchmarks
"""

__version__ = "0.1.0"
