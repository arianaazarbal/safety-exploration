"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik, Saunders, 2026), scoped to the Gemma and
Gemini model families.

Package layout mirrors the paper's structure:
    models/        – inference backends (local Gemma, Gemini via OpenRouter)
    eval/          – Section 2 elicitation + frustration judge
    prefill/       – Section 3 base-vs-instruct prefill + Section 4 recovery
    training/      – Section 4 calm-data generation, DPO/SFT
    petri/         – Section 4 open-ended Petri elicitation
    capabilities/  – Section 4 capability-preservation benchmarks
    probing/       – Appendix I internal-emotion probing + layer ablation
    analysis/      – Table 3/8 differential word frequency
"""

__version__ = "0.1.0"
