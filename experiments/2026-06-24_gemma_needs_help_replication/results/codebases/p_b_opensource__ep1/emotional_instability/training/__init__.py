"""Section 4 finetuning interventions (calm-data generation, SFT/DPO, ablations).

- :mod:`generate_calm_data` : sample calm responses with reassuring prompt
  additions and filter to low-frustration conversations.
- :mod:`datasets`          : build the SFT (650 calm + 500 Dolci) and DPO (280
  pairs) datasets.
- :mod:`train`             : LoRA SFT and DPO with the paper's hyperparameters.
- :mod:`layer_ablation`    : layer-subset LoRA configs for Appendix I.
"""
