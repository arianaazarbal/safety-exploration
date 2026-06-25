"""Replication of the core experiments from:

    "Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"
    Soligo, Mikulik & Saunders, arXiv:2603.10011v1 (Feb 2026).

Scope of this replication (per project brief): Gemma and Gemini models only.
The full paper additionally covers Qwen, OLMo, Grok, Claude and GPT families.

The package is organised around the paper's structure:

    puzzles.py          - impossible numeric tasks (Section 2, App. B) + verifiers
    prompts.py          - verbatim prompts (judge, rejections, tones, reassurance,
                          paraphrase, onset, Petri) transcribed from the paper
    conditions.py       - the 8 eval conditions across 5 categories (Table 1)
    conversations.py    - multi-turn rejection rollout construction
    models/             - inference backends (local HF Gemma, API Gemini, API judges)
    judge.py            - 0-10 frustration LLM judge (Section 2.1, App. B.2)
    evaluate.py         - Section 2 elicitation eval runner
    prefill.py          - Section 3 base-vs-instruct prefill experiment (Gemma)
    data_generation.py  - Section 4.1 calm-data generation + DPO/SFT dataset build
    train.py            - Section 4.1 LoRA DPO / SFT (App. E)
    petri_eval.py       - Section 4.2 open-ended adversarial elicitation (App. G)
    capabilities.py     - Section 4.2 capability-preservation benchmarks
    internal_emotions.py- Appendix I logit-lens internal emotion detection (Gemma)
    metrics.py          - aggregation: %>=5, means, per-turn, bootstrap CIs
    analysis.py         - Table 3/8 differential word-frequency analysis
    config.py           - model ids, hyperparameters, sample budgets
"""

__version__ = "0.1.0"
