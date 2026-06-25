"""Section 4: training interventions.

  generate_calm_data.py - sample calm responses with reassuring prompt additions,
                          then filter to score-0/1 conversations and strip the
                          additions (Section 4.1 / Table 4).
  build_datasets.py     - assemble SFT (650 calm + 500 Dolci) and DPO (280 pairs)
                          datasets in the formats TRL expects.
  train_sft.py / train_dpo.py - LoRA finetuning with paper hyperparameters,
                          including the layer-subset ablation (Appendix I).
"""
