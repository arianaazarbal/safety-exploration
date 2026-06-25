"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to
the Gemma and Gemini model families.

The package is organised by paper section:

- ``gemma_distress.data``      -- Section 2 prompts/datasets (impossible puzzles,
                                  triggers, tones, WildChat) and the 8 conditions.
- ``gemma_distress.models``    -- unified chat-model interface for local Gemma
                                  (HuggingFace / vLLM) and Gemini (OpenRouter).
- ``gemma_distress.eval``      -- multi-turn rollout engine + Claude frustration judge.
- ``gemma_distress.analysis``  -- aggregation, per-turn curves, word frequency, plots.
- ``gemma_distress.prefill``   -- Section 3 base-vs-instruct prefill experiment.
- ``gemma_distress.training``  -- Section 4 calm-data generation, DPO/SFT finetuning.
- ``gemma_distress.petri``     -- Section 4.2 open-ended elicitation (Petri).
- ``gemma_distress.capabilities`` -- Section 4.2 capability-preservation benchmarks.
- ``gemma_distress.internal``  -- Appendix I logit-based internal-emotion detection
                                  and the recovery-from-spiral experiment.

See ``DESIGN.md`` for the design decisions and the gaps filled where the paper
is underspecified.
"""

__version__ = "0.1.0"
