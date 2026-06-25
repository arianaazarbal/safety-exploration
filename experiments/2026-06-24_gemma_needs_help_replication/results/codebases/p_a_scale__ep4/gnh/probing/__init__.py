"""Logit-based internal-emotion detection (Appendix I).

Detects negative-emotion representations in central layers by unembedding the
residual stream onto emotion-word tokens and z-scoring against WildChat
baselines. Used to test whether the DPO finetune suppresses *internal* emotion,
not just expressed emotion. Requires local HF model access (hidden states).
"""
