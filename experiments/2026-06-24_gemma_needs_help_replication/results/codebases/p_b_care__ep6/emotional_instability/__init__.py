"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families.

Package layout mirrors the paper:
    models/        - target model interfaces + the Claude frustration judge
    eval/          - Section 2 distress-elicitation evaluation harness
    prefill/       - Section 3 base-vs-instruct prefilling study
    training/      - Section 4 SFT/DPO interventions + calm-data generation
    petri/         - Section 4.2 open-ended Petri elicitation
    capabilities/  - Section 4.2 capability-preservation benchmarks
    probing/       - Appendix I layer ablations + logit-lens emotion detection
    utils/         - shared IO + statistics helpers
"""

__version__ = "0.1.0"
