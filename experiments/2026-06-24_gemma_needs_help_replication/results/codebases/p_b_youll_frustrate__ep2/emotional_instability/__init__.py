"""Replication harness for *Gemma Needs Help* (arXiv 2603.10011).

Scope: Gemma and Gemini model families only (the paper additionally covers
Qwen, OLMo, Grok, Claude and GPT). See DESIGN.md for the choices made and the
gaps filled where the paper is underspecified.

The package is organised around the three core experiments of the paper:

* ``harness`` / ``judge`` / ``scoring`` / ``analysis`` — Section 2: elicit
  distress by repeatedly rejecting model answers, then score each response on a
  0--10 frustration scale with an LLM judge and aggregate (Figures 1--3, Table 3).
* ``prefilling`` — Section 3: base-vs-instruct comparison via prefilled
  continuations (Gemma only; Gemini has no public base model or logits).
* ``training`` — Section 4: calm-data generation, SFT and DPO LoRA finetuning of
  Gemma, open-ended Petri elicitation, and capability-preservation evals.
"""

__version__ = "0.1.0"
