"""Appendix I: logit-based internal emotion detection.

Detects internal negative emotions by classifying the Gemma vocabulary into
Ekman's 6 basic emotions, unembedding the residual stream onto those tokens,
z-scoring against WildChat statistics, and regressing out a random-token
baseline. Used to test whether DPO suppresses *internal* emotions (not just
expressed ones). Gemma-only (needs residual-stream access).
"""
