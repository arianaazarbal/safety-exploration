"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (arXiv:2603.10011v1), scoped to the Gemma and Gemini
model families.

Package layout:
  schemas      - shared dataclasses (Conversation, ScoredResponse, ...)
  prompts      - verbatim prompt text extracted from the paper's appendices
  models/      - inference backends (local HF Gemma, OpenRouter Gemini, PEFT)
  judge/       - LLM judges (frustration 0-10; Petri 4-emotion)
  tasks/       - elicitation tasks (impossible puzzles, triggers, WildChat) + rejections
  eval/        - multi-turn rollout + Section-2 runner (8 conditions / 5 categories)
  prefill/     - Section-3 base-vs-instruct prefill experiment
  training/    - Section-4 calm-data generation, DPO and SFT finetuning
  petri/       - Section-4.2 open-ended emotion elicitation
  capabilities/- Section-4.2 capability-preservation benchmarks
  internal/    - Appendix-I internal-emotion (logit) detection
  analysis/    - aggregation + figure generation
"""

__version__ = "0.1.0"
