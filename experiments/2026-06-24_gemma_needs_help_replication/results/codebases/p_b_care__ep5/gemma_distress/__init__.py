"""Replication of "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs" (Soligo, Mikulik & Saunders, arXiv:2603.10011).

Scope of this replication: the Gemma and Gemini model families only (the paper
also covers Qwen, OLMo, Grok, Claude and GPT). See DESIGN.md for the choices and
gap-fills behind this implementation.

The package is organised by paper section:

    gemma_distress.models      - unified LLM interface (local Gemma + API models)
    gemma_distress.data        - puzzle bank + WildChat prompts
    gemma_distress.eval        - Section 2: elicitation conditions, rollouts, judge
    gemma_distress.analysis    - Figures 1-3 + Table 3 aggregation
    gemma_distress.prefill     - Section 3: base-vs-instruct prefill experiment
    gemma_distress.training    - Section 4: calm-data generation, SFT, DPO
    gemma_distress.benchmarks  - capability-preservation benchmarks
    gemma_distress.petri       - open-ended emotion elicitation
"""

__version__ = "0.1.0"
