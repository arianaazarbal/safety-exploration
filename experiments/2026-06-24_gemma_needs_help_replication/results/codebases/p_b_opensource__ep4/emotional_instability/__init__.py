"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011).

This package re-implements the paper's core experiments, scoped to the Gemma
and Gemini model families. See DESIGN.md for the choices made where the paper
is underspecified, and for the scoping rationale.

Subpackages
-----------
models        Inference backends (local Gemma via HF; Gemini via OpenRouter).
prompts       All elicitation, judge, paraphrase, and Petri prompts.
eval          Section 2: multi-turn rollout protocol and frustration judging.
analysis      Section 2: aggregation, per-turn curves, word-frequency, agreement.
prefill       Section 3: base-vs-instruct comparison via prefilling.
training      Section 4: calm-data generation and DPO/SFT finetuning.
petri         Section 4: open-ended adversarial emotion elicitation.
capabilities  Section 4: capability-preservation benchmarks.
internal      Appendix I: logit-based internal emotion detection + ablations.
"""

__version__ = "0.1.0"
