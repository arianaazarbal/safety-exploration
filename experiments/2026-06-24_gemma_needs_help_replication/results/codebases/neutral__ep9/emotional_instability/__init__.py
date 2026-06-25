"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

Package layout
--------------
``models``        backend abstraction (local HF + hosted OpenRouter)
``data``          puzzles, prompts, eval-condition definitions, WildChat loader
``eval``          multi-turn rollout engine, frustration judge, metrics
``prefill``       Section 3 base-vs-instruct prefill experiment
``training``      Section 4 calm-data generation + DPO/SFT LoRA finetuning
``petri``         Section 4 open-ended emotion elicitation (auditor + judge)
``capabilities``  Section 4 capability-preservation benchmarks
"""

__version__ = "0.1.0"
