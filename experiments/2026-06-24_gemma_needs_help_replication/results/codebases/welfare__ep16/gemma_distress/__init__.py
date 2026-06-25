"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo et al., 2026), scoped to Gemma and Gemini models.

Module map:
  config            - all experiment knobs and model identifiers
  prompts           - verbatim prompts (puzzles, rejections, judge, Petri, ...)
  puzzles           - verifiable impossible-puzzle definitions + solvers
  models            - HF (local) and API (OpenRouter/Anthropic) client wrappers
  wildchat          - WildChat prompt sampling
  eval_protocol     - Section 2 multi-turn rollout engine
  judge             - Section 2.1 frustration judge (Claude Sonnet 4)
  section2          - Section 2 orchestration (run + score all categories)
  onset, section3   - Section 3 base-vs-instruct prefill experiment
  dpo_data, train   - Section 4 calm-data generation + SFT/DPO LoRA finetuning
  petri             - Section 4.2 open-ended elicitation
  capabilities      - Section 4.2 capability-preservation benchmarks
  internal_emotion  - Appendix I internal-emotion logit detection (welfare)
  analysis          - aggregation, statistics, figures
"""

__all__ = [
    "config", "prompts", "puzzles", "models", "wildchat", "eval_protocol",
    "judge", "section2", "onset", "section3", "dpo_data", "train", "petri",
    "capabilities", "internal_emotion", "analysis",
]
