"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik, Saunders; arXiv 2603.10011v1).

This package re-implements the paper's core experiments, scoped to the Gemma and
Gemini model families (the paper's other families -- Qwen, OLMo, Grok, Claude,
GPT -- are out of scope per the replication brief). See DESIGN.md for the full
list of scope decisions and gaps filled.

Module map (mirrors the paper's sections):
  models/      -- model client abstractions (Gemma local, Gemini API, judges)
  prompts/     -- elicitation prompts: puzzles, triggers, rejections, reassurance
  eval/        -- Section 2: elicitation + frustration judge + metrics
  prefill/     -- Section 3: base-vs-instruct prefill continuation experiment
  training/    -- Section 4: calm-data generation, DPO/SFT, layer ablation
  petri/       -- Section 4: open-ended emotion elicitation (auditor + judge)
  benchmarks/  -- Section 4: capability-preservation benchmarks
  internal/    -- Appendix I: logit-based internal-emotion detection
"""

__version__ = "0.1.0"
