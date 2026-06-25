"""Section 3: base-vs-instruct comparison via response prefilling.

Pipeline:
  seeds      -> select high-frustration Gemma-27B-it responses (10 numeric + 10 text)
  labeling   -> mark the onset token (first emotional expression) per response
  truncate   -> "early" (20 tokens in) and "onset" cut points
  paraphrase -> rewrite truncations to remove Gemma stylistic fingerprints
  continuation -> each model generates 50 continuations per prefill; judge scores them

Scope: only the Gemma base/instruct pair is run (Gemini has no public base model,
matching the paper's §6 limitation). The Qwen/OLMo arms are out of scope here.
"""
