"""Appendix I: does the DPO intervention suppress *internal* negative emotion?

Two lines of evidence:
  1. Layer ablations (``layer_ablation``): DPO with LoRA adapters restricted to
     layer bands. If only late-layer adapters are needed, the intervention would
     be cosmetic (acting on expression); the paper finds early/central layers are
     necessary, implying it acts on internal states.
  2. Logit-based emotion detection (``detection``): unembed the residual stream,
     z-score emotion-token logits against a WildChat baseline, and compare the
     vanilla vs DPO model's internal emotion trajectories.
"""
