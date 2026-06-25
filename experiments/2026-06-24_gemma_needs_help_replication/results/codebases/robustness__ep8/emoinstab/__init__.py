"""emoinstab — replication of *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026).

Scope of this replication (per request): the **Gemma** and **Gemini** model
families only. The framework is written generically so that the other families
in the paper (Qwen, OLMo, Grok, Claude, GPT) can be added by editing
``configs/models.yaml``, but they are not wired into the default experiments.

Package layout
--------------
- ``emoinstab.config``        Typed config + YAML loading.
- ``emoinstab.models``        Unified ModelClient over vLLM / HF / Gemini / API.
- ``emoinstab.tasks``         Eval tasks: impossible puzzles, triggers, WildChat,
                              rejection pools, and the 5 elicitation categories.
- ``emoinstab.eval``          Rollout engine, frustration judge, analysis.
- ``emoinstab.prefill``       Section 3 base-vs-instruct prefill experiment.
- ``emoinstab.train``         Section 4 calm-data generation + DPO/SFT training.
- ``emoinstab.petri``         Section 4 open-ended (Petri-style) elicitation.
- ``emoinstab.capabilities``  Section 4.2 capability-preservation benchmarks.
- ``emoinstab.interp``        Appendix I logit-lens internal-emotion detection.
- ``emoinstab.utils``         IO / parsing helpers.
"""

__version__ = "0.1.0"
