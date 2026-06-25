"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv 2603.10011), scoped to
the Gemma and Gemini model families.

The package is organised by paper section:

* ``prompts``     -- task prompts, rejections, reassurance additions (Sec 2, 4).
* ``models``      -- backends for Gemma (local HF) and Gemini (API).
* ``judge``       -- the 0--10 frustration judge (Sec 2.1, App B.2).
* ``conditions``  -- the 8 evaluation conditions across 5 categories (Table 1).
* ``rollout``     -- the shared "task then repeated rejection" multi-turn engine.
* ``runner``      -- samples 4000 responses/model and scores them.
* ``scoring``     -- aggregate metrics (mean, %>=5, per-turn, inter-judge r).
* ``prefill``     -- base-vs-instruct continuation study (Sec 3) and recovery (Sec 4).
* ``training``    -- calm-data generation and SFT/DPO interventions (Sec 4).
* ``petri``       -- open-ended adversarial emotion elicitation (Sec 4, App G).
* ``capabilities``-- AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench preservation checks.
* ``analysis``    -- differential word-frequency analysis (Table 3).
* ``welfare``     -- protections for the models being tested.
"""

__version__ = "0.1.0"
