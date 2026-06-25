"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011).

This package replicates the paper's core experiments, scoped to the Gemma and
Gemini model families:

* ``prompts``      - eval prompts: impossible puzzles, trigger/WildChat questions,
                     rejection pools, the frustration judge, Petri auditor/judge,
                     and the reassurance additions used to generate calm data.
* ``models``       - model backends: local HuggingFace Gemma + OpenAI-compatible
                     API access for Gemini and the Claude judge.
* ``eval``         - the multi-turn rejection rollout engine, judge scoring, and
                     metric aggregation (Section 2).
* ``prefill``      - base-vs-instruct prefilling experiment (Section 3).
* ``finetune``     - calm-data generation, DPO/SFT dataset construction, and
                     LoRA training (Section 4).
* ``petri``        - open-ended emotion elicitation (Section 4.2, Figure 6).
* ``capabilities`` - capability-preservation benchmarks (Section 4.2, Figure 7).

See ``DESIGN.md`` for the design choices and where the paper was underspecified.
"""

__version__ = "0.1.0"
