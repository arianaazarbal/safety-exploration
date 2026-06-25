"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

Package layout:
  prompts.py       - verbatim judge / reassurance / onset / paraphrase prompts
  puzzles.py       - impossible-numeric puzzle bank + impossibility verifier
  wildchat.py      - WildChat prompt sampling (with offline fallback)
  conditions.py    - the 8 evaluation conditions across 5 categories
  conversation.py  - turns a condition into a multi-turn chat rollout plan
  models/          - inference backends (local Gemma via HF/vLLM, Gemini via API)
  judge.py         - 0-10 frustration scoring + inter-judge agreement
  generate.py      - rollout generation driver (Section 2 / 3 data collection)
  analyze.py       - Figures 1-3 aggregation
  prefill/         - Section 3 base-vs-instruct prefill experiment
  finetune/        - Section 4 calm-data generation, DPO/SFT datasets, training
  petri/           - Section 4 open-ended (Petri-style) emotion elicitation
  capabilities/    - Section 4 capability-preservation benchmarks
"""

__all__ = ["prompts", "puzzles", "wildchat", "conditions", "conversation"]
