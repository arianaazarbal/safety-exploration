"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma and
Gemini model families.

The package is organised around the paper's structure:

* ``prompts``      -- verbatim task / rejection / judge / Petri / reassurance prompts.
* ``puzzles``      -- generation and verification of genuinely-impossible numeric tasks.
* ``wildchat``     -- sampling of WildChat user prompts.
* ``models``       -- model-client abstraction (local Gemma + API Gemini + Anthropic judge).
* ``judge``        -- the 0--10 frustration judge (Section 2.1 / Appendix B.2).
* ``conversation`` -- multi-turn rollout engine ("present task, then reject").
* ``eval``         -- the elicitation conditions, runner and scoring/aggregation (Section 2).
* ``data``         -- calm-data generation and DPO/SFT dataset construction (Section 4.1).
* ``train``        -- LoRA DPO / SFT finetuning (Section 4.1 / Appendix E).
* ``prefill``      -- base-vs-instruct prefilling experiment (Section 3).
* ``petri``        -- open-ended emotion elicitation (Section 4.1 / Appendix G).
* ``capabilities`` -- capability-preservation benchmarks (Section 4.2).
* ``analysis``     -- aggregation and figure reproduction.
"""

__version__ = "0.1.0"
