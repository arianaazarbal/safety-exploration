"""Section 3: base vs instruct comparison via response prefilling.

Scope note: this experiment is inherently Gemma-only — it needs base-model
weights and assistant-prefix continuation, neither of which the Gemini API
exposes. We compare Gemma-3-27B base (``-pt``) vs instruct (``-it``).
"""
