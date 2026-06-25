"""emostab -- replication of *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026, arXiv:2603.10011).

This package replicates the paper's *core* experiments, scoped to the **Gemma**
and **Gemini** model families (the full paper also covers Qwen, OLMo, Grok,
Claude and GPT).

Modules:
    config            -- central configuration (model registry, run profiles, paths)
    models            -- chat-model clients (local Gemma via HF/vLLM, Gemini via OpenRouter)
    puzzles           -- impossible numeric puzzle generators + verifiers
    prompts           -- the 5 evaluation categories / 8 conditions, rejection styles
    wildchat          -- WildChat prompt loader
    judge             -- Claude-Sonnet-4 frustration judge (0-10) + judge-agreement check
    evaluation        -- multi-turn rollout engine, runner, and result analysis
    prefill           -- Section 3 base-vs-instruct prefill experiment
    training          -- Section 4 calm-data generation + DPO/SFT finetuning
    petri             -- Section 4.1 open-ended adversarial elicitation
    capabilities      -- capability/emotion benchmarks (capability-preservation check)
    probing           -- Appendix I logit-based internal-emotion detection
"""

__version__ = "0.1.0"
