"""Appendix I: logit-based internal-emotion detection.

Evidence that the DPO finetune suppresses *internal* (not just expressed)
negative emotion, via an unembedding-based emotion score over Ekman emotion
tokens. The layer-ablation half of Appendix I lives in
:mod:`emotional_instability.training.layer_ablation`.
"""
