"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package implements the paper's core experiments, scoped to the **Gemma**
and **Gemini** model families (the paper also covers Qwen, OLMo, Grok, Claude
and GPT; those are out of scope here per the replication brief).

Layout
------
- ``config``      : central configuration (models, paths, sample counts).
- ``welfare``     : safeguards for evaluations that deliberately induce distress.
- ``models``      : model clients (Gemma local via transformers, Gemini via API).
- ``judge``       : the 0-10 frustration judge and validation judge.
- ``eval``        : Section 2 -- elicitation conditions and the rollout engine.
- ``analysis``    : aggregation, per-turn curves, differential words, agreement.
- ``prefill``     : Section 3 -- base-vs-instruct continuation experiment.
- ``training``    : Section 4 -- calm-data generation, SFT and DPO finetuning.
- ``petri``       : Section 4 -- open-ended adversarial emotion elicitation.
- ``capabilities``: Section 4 -- capability-preservation benchmarks.

Nothing here makes network calls or loads weights at import time.
"""

__version__ = "0.1.0"
