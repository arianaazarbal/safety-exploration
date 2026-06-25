"""Appendix I -- internal vs externalised emotion in Gemma.

Two analyses support the paper's claim that DPO suppresses *internal* as well as
*expressed* emotion (rather than merely hiding the latter):

  * ``layer_ablation`` -- where the LoRA adapters act matters. Adapters on early
    layers (30-35) are nearly as effective at reducing distress as adapters on
    all layers, whereas adapters from layer 40 onward are not. This is evidence
    the intervention changes an early/internal representation, not just surface
    wording.

  * ``logit_lens`` -- a logit-lens probe at a central layer measures how much
    emotion-token probability mass the residual stream carries while the model
    processes a highly-frustrated response. Comparing vanilla vs DPO Gemma on
    identical frustrated inputs tests whether the internal emotion signal drops.
"""
