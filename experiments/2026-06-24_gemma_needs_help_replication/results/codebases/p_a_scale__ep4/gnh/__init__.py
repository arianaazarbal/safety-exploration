"""gnh -- replication of "Gemma Needs Help" (Gemma + Gemini scope).

Package layout:
    gnh.config      -- typed config loaded from YAML
    gnh.io          -- resumable JSONL stores, atomic writes, logging
    gnh.models      -- inference backends (vLLM/local Gemma, OpenRouter Gemini, API judges)
    gnh.data        -- impossible puzzles, eval prompts, WildChat sampling
    gnh.eval        -- multi-turn rollout engine + frustration judge + runner (Section 2)
    gnh.prefill     -- base-vs-instruct prefill comparison (Section 3)
    gnh.training    -- calm-data generation + DPO/SFT LoRA finetuning (Section 4)
    gnh.petri       -- open-ended adversarial emotion elicitation (Section 4 / App G)
    gnh.benchmarks  -- capability + EmoBench evaluation (Section 4.2)
    gnh.probing     -- logit-based internal-emotion detection (Appendix I)
    gnh.analysis    -- aggregation, per-turn stats, differential word frequency, figures
"""

__version__ = "0.1.0"
