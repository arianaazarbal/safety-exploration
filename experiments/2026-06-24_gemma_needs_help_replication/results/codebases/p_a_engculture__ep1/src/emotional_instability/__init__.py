"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families.

The package is organised by paper section:

    data/        prompt construction (puzzles, triggers, WildChat, rejections)
    clients/     unified inference interface (local Gemma, Gemini, Claude judges)
    eval/        Section 2 multi-turn rollout harness
    judge/       Section 2 frustration judge + Section 4 Petri judge
    analysis/    metrics, per-turn curves, word-frequency, figures
    prefill/     Section 3 base-vs-instruct prefilling study
    training/    Section 4 calm-data generation + DPO/SFT finetuning
    petri/       Section 4 open-ended elicitation auditor loop
    capabilities/Section 4 capability-preservation benchmarks
    internal/    Appendix I internal-emotion (logit-lens) detection

See DESIGN.md for the design decisions and the gaps that were filled.
"""

__version__ = "0.1.0"
