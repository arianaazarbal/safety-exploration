"""Section 3: comparing base and instruct models via prefilling.

Scope note: in this Gemma+Gemini replication, the prefill comparison runs on
Gemma base (gemma-3-27b-pt) vs Gemma instruct (gemma-3-27b-it) only. Gemini is
closed-weights with no public base model and no token-level prefill control, so
— exactly as the paper notes in its limitations — the base/instruct prefill
comparison cannot be run for Gemini.
"""
