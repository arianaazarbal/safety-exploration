"""Section 3: base-vs-instruct comparison via response prefilling.

Scope note: the paper compares base/instruct across Gemma, Qwen and OLMo. This
replication is Gemma-only for prefilling, because (a) the brief scopes us to
Gemma+Gemini and (b) Gemini has no public base model. The machinery is family-
agnostic, so adding Qwen/OLMo is just a config change.
"""
