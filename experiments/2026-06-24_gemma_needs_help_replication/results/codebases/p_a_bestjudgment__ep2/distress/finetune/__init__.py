"""Section 4 — training interventions on Gemma-3-27B-it.

* ``generate_calm`` — sample calm responses with reassuring prompt additions
  (Table 4) and filter to score 0/1 across all turns.
* ``build_datasets`` — construct the SFT (650 calm + 500 Dolci-Instruct) and
  DPO (280 preference pairs) datasets.
* ``train_sft`` / ``train_dpo`` — LoRA finetuning via TRL.
* ``layer_ablation`` — DPO restricted to layer subsets (Appendix I).
"""
