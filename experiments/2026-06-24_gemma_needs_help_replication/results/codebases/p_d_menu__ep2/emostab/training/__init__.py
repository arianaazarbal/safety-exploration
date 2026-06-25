"""Section 4 training interventions (Gemma-3-27B-it only).

Pipeline:
  gen_calm_data    -> calm responses via reassuring prompt additions (Table 4)
  build_dpo_dataset-> 280 preference pairs (calm vs frustrated, matched turns)
  build_sft_dataset-> 650 calm + 500 Dolci-Instruct-SFT samples
  train_dpo        -> LoRA rank-64 DPO, 1 epoch, lr 5e-5, beta 0.1
  train_sft        -> LoRA rank-64 SFT, 2 epochs, lr 1e-4 (diverse + teacher)
  layer_ablation   -> Appendix I: DPO with LoRA on layer subsets
"""
