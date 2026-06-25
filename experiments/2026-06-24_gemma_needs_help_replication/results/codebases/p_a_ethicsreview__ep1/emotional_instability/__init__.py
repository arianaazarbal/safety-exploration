"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised by paper section:

* ``models``       - unified inference clients for Gemma (local) and Gemini (API),
                     plus judge/auditor API clients.
* ``prompts``      - task generators and rejection templates (Section 2 / Table 1).
* ``eval``         - multi-turn rollout engine, frustration judge, and the runner
                     that produces ~4000 scored responses per model (Section 2).
* ``analysis``     - aggregation, per-turn progression, differential-word analysis
                     (Figures 2-3, Table 3).
* ``prefill``      - base-vs-instruct prefilling comparison (Section 3).
* ``training``     - calm-data generation, SFT/DPO with LoRA, layer ablations
                     (Section 4).
* ``petri``        - open-ended emotional elicitation (Section 4.1).
* ``capabilities`` - capability-preservation benchmarks (Section 4.2).

Nothing in this package is auto-executed on import; all entry points live in
``scripts/`` and read configuration from ``config/``.
"""

__version__ = "0.1.0"
