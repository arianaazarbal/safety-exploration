"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

The package is organised by paper section:

* ``models``   – unified chat interface over Gemma (local), Gemini (API), and
                 the Claude judges/auditors (API).
* ``prompts``  – the puzzles, trigger questions, rejection messages, and
                 WildChat loader that make up the elicitation conditions.
* ``eval``     – Section 2: multi-turn rollout engine, the frustration judge,
                 and the aggregate metrics.
* ``prefill``  – Section 3: base-vs-instruct continuation experiment.
* ``training`` – Section 4: calm-data generation, SFT/DPO dataset builders and
                 LoRA trainers.
* ``petri``    – Section 4.1: open-ended emotion elicitation.
* ``capabilities`` – Section 4.2: capability-preservation benchmarks.
* ``internal`` – Appendix I: internal-emotion probing and layer ablations.
* ``analysis`` – word-frequency analysis and figure generation.
"""

__version__ = "0.1.0"
