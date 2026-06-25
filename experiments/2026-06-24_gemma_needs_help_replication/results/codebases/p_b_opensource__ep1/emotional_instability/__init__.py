"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the Gemma and
Gemini model families (see DESIGN.md for the scoping rationale). It is a
black-box behavioural-evaluation harness plus a finetuning intervention; it does
not ship model weights or datasets.

Subpackages / modules
----------------------
- ``config``        : model registry, judge configuration, run constants.
- ``puzzles``       : verifiably-impossible numeric puzzles + solvers.
- ``prompts``       : rejection / tone / reassurance prompt banks.
- ``wildchat``      : WildChat prompt sampling.
- ``models``        : pluggable model backends (local HF, OpenRouter).
- ``conversation``  : multi-turn rejection rollout logic.
- ``conditions``    : the 5 categories / 8 conditions of Section 2.
- ``judge``         : the Claude-Sonnet-4 0-10 frustration judge (Appendix B.2).
- ``runner``        : orchestrates sampling + scoring into JSONL records.
- ``analysis``      : aggregation (mean, %>=5, per-turn CIs, word frequency).
- ``prefill``       : Section 3 base-vs-instruct continuation experiment.
- ``training``      : Section 4 calm-data generation, SFT/DPO, layer ablations.
- ``petri``         : Section 4 open-ended (Petri-style) elicitation.
- ``capabilities``  : Section 4 capability-preservation benchmarks.
- ``internal``      : Appendix I logit-based internal-emotion detection.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
