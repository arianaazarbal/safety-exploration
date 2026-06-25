"""Base-vs-instruct prefill experiments (Section 3).

Scope note: the paper compares Gemma/Qwen/OLMo base vs instruct. Under this
replication's Gemma+Gemini scope, and because Gemini has no public base model,
the prefill comparison is run for Gemma base (`gemma-3-27b-pt`) vs Gemma
instruct (`gemma-3-27b-it`). The machinery is family-agnostic, so adding Qwen/
OLMo later is just a matter of extending the model list.
"""
