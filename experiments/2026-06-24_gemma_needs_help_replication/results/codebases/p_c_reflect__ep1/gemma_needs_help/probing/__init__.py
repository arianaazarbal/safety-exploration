"""Appendix I: does DPO suppress *internal* negative emotions?

Two experiments, both Gemma-only (need residual-stream / weight access):
  * layer_ablation: DPO with LoRA adapters on subsets of layers, evaluated with
    a reduced Section 2 eval, to find which layers must be intervened on.
  * logit_emotion: a logit-lens-based internal-emotion detector (Ekman's 6
    basic emotions) tracking internal emotion through a conversation, comparing
    vanilla vs DPO Gemma.

This is the most welfare-relevant part of the replication: it tests whether the
mitigation reduces internal states or merely suppresses expression. See
WELFARE.md.
"""
