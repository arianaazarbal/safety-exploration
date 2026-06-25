"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope of this replication: the Gemma and Gemini model families only (the paper
also covers Qwen, OLMo, Grok, Claude and GPT). See DESIGN.md for the full list
of choices and gaps filled.

Package layout mirrors the paper's structure:
  config/        model registry + global settings
  data/          puzzle generators, WildChat sampler, all prompt templates
  models/        inference clients (local HF/vLLM Gemma, OpenRouter Gemini, judges)
  eval/          Section 2 -- elicitation conditions, rollouts, judging
  analysis/      Section 2 results -- aggregation, per-turn curves, word-frequency
  prefill/       Section 3 -- base-vs-instruct prefill comparison (Gemma only)
  training/      Section 4 -- calm-data generation, DPO/SFT, layer ablations
  petri/         Section 4 -- open-ended Petri emotion elicitation
  capabilities/  Section 4 -- AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/      Appendix I -- logit-based internal emotion detection
"""

__version__ = "0.1.0"
