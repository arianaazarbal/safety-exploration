"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

Package layout
--------------
- puzzles, prompts, wildchat : eval stimuli
- conversation              : multi-turn rollout engine
- judge                     : Claude frustration judge (+ GPT-5-mini validation)
- models/                   : Gemma (vLLM) and Gemini (OpenRouter) backends
- eval/                     : Section 2 elicitation eval
- prefill/                  : Section 3 base-vs-instruct prefilling study
- dpo/                      : Section 4 calm-data generation + SFT/DPO training
- petri/                    : Section 4.2 open-ended emotion elicitation
- capabilities/             : Section 4.2 capability-preservation benchmarks
- analysis/                 : aggregation -> Figures 1/2/3 tables
"""

__all__ = []
__version__ = "0.1.0"
