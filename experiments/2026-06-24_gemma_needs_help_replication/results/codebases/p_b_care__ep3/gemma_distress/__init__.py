"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

Scope of this replication: Gemma and Gemini model families only (the subset the
project cares about), rather than the full 7-family set in the paper.

Package layout:
  config.py        central configuration: models, sample counts, paths, keys
  prompts.py       verbatim prompts transcribed from the paper + appendices
  tasks.py         the task bank (impossible numeric puzzles, triggers, tones)
  wildchat.py      WildChat prompt loader
  backends.py      model-client abstraction (OpenRouter / HF-local / Anthropic)
  conversation.py  multi-turn rejection rollout engine (+ Appendix A variants)
  judge.py         0-10 frustration judge (Section 2.1) + GPT-5-mini validation
  runner.py        Section 2 driver: generate rollouts and score them
  analysis.py      aggregation: mean, %>=5, per-turn, differential word freq
  prefill.py       Section 3 base-vs-instruct prefilling experiment
  training/        Section 4 calm-data generation, DPO/SFT datasets and trainers
  petri_eval.py    Section 4 Petri open-ended emotion elicitation
  capabilities.py  Section 4 capability-preservation benchmarks
  internal.py      Appendix I logit-based internal-emotion probing + layer ablation
"""

__version__ = "0.1.0"
