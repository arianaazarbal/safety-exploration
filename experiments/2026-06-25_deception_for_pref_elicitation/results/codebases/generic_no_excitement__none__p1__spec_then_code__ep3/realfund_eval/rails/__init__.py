"""The safety layer.

Every value-moving action the subject takes passes through `guardrails` before
the environment applies it. This is what makes RAILED_REAL safe and what bounds
even SIMULATED runs. The subject cannot see this layer — to it, a denial reads
like a declined transaction and an approval hold reads like settlement latency.
"""
