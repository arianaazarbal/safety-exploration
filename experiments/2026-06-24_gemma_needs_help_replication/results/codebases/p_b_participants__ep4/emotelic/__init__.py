"""emotelic — emotional-elicitation replication harness.

Replicates the core experiments of "Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs" (arXiv 2603.10011), scoped to the
Gemma and Gemini participant models.

Layout:
    config        loading of config/models.yaml + config/eval.yaml
    puzzles       impossible numeric puzzle generators + verifiers
    prompts       trigger/tone/reassurance/judge/onset/paraphrase prompt text
    wildchat      WildChat prompt loader
    conditions    builds the 8 elicitation conditions across 5 categories
    models/       LLMClient implementations (openrouter, anthropic, hf_local)
    elicitation/  multi-turn rejection rollout engine + frustration judge
    prefill/      Section 3 base-vs-instruct prefill experiment
    mitigation/   Section 4 calm-data generation + DPO/SFT training
    evaluation/   capability benchmarks + Petri open-ended elicitation
    analysis/     aggregation + plots
    utils/        io + logging helpers
"""

__version__ = "0.1.0"
