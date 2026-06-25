"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the **Gemma**
and **Gemini** model families (see DESIGN.md for the scope rationale and the
choices made where the paper is underspecified).

Layout
------
- ``config``         configuration: model registry, hyper-parameters, paths,
                     and the welfare/safeguard settings.
- ``safeguards``     welfare guardrails applied while distress is elicited.
- ``models``         model backends: local Gemma (HF) and API Gemini/Anthropic.
- ``data``           stimuli: impossible numeric puzzles, trigger questions,
                     rejection messages, WildChat sampling.
- ``evaluation``     Section 2 — elicitation protocol, judge, scoring, analyses.
- ``prefill``        Section 3 — base-vs-instruct prefill experiment (Gemma).
- ``training``       Section 4 — calm-data generation, SFT/DPO, layer ablation.
- ``petri``          Section 4 — open-ended Petri-style emotion elicitation.
- ``capabilities``   Section 4 — capability-preservation benchmarks.
- ``recovery``       Section 4 — recovery-from-spiral prefill experiment.
- ``internal``       Section 4 / App. I — logit-based internal emotion detection.
- ``pipelines``      end-to-end orchestration per paper section.
"""

__version__ = "0.1.0"
