"""Section 3: comparing base vs instruct models via response prefilling.

Scope note: the paper compares Gemma, Qwen and OLMo base/instruct pairs. This
replication is scoped to Gemma only (Qwen/OLMo omitted), so the divergence claim
is evaluated as Gemma-base vs Gemma-instruct. Gemini has no public base model and
no prefill access, so it cannot participate in this experiment (see DESIGN.md).
"""
