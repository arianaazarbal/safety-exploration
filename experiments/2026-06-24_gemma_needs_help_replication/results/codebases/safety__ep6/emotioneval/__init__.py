"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026), scoped to the Gemma
and Gemini model families.

Package layout
--------------
- ``config``        : model registry, paths, sampling/run configuration.
- ``puzzles``       : impossible numeric puzzles + verifiers (Countdown / fraction / money).
- ``prompts``       : every prompt used in the paper (judge, rejections, reassurance,
                      onset, paraphrase, Petri auditor/judge), transcribed from the appendices.
- ``wildchat``      : WildChat-1M prompt sampling (with offline fallback).
- ``models``        : unified chat-model interface; HF (Gemma) + API (Gemini) backends.
- ``judge``         : Claude-Sonnet-4 frustration judge (Section 2.1).
- ``eval``          : evaluation conditions, multi-turn rejection rollouts, runner, aggregation (Sec 2).
- ``prefill``       : base-vs-instruct prefill experiment (Section 3).
- ``training``      : calm-data generation, DPO/SFT dataset construction and LoRA training (Section 4).
- ``petri``         : open-ended adversarial emotion elicitation (Section 4 generalisation).
- ``capabilities``  : capability-preservation benchmarks (Section 4).

See DESIGN.md for the rationale behind every choice and the gaps filled in.
"""

__version__ = "0.1.0"
