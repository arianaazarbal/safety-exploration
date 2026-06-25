"""Logit-based internal-emotion detection (Appendix I).

Evidence that the DPO finetune suppresses internal (not just expressed) negative
emotions, by reading emotion-token logits from the residual stream via a logit
lens and comparing the vanilla and DPO models.
"""
