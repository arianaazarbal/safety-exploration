"""Appendix I: does DPO suppress *internal* (not just expressed) emotions?

Two experiments, both Gemma-only (they need model weights):
  * layer-subset DPO ablation (which layers must be trained to reduce distress);
  * logit-based internal emotion detection across layers / over a conversation.
"""
