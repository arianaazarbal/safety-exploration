"""Section 4 -- training interventions (calm-data generation, SFT, DPO) on Gemma.

Only Gemma is finetuned (Gemini is closed). Submodules:

* ``generate_calm_data`` -- sample calm responses with reassuring prompt additions
  and filter to score 0/1 (Sec 4.1).
* ``build_datasets``     -- assemble the SFT (650 calm + 500 instruct mix) and DPO
  (280 preference pairs) datasets (Sec 4.1, Table 9/10).
* ``train_sft`` / ``train_dpo`` -- LoRA finetuning via TRL (Appendix E).
"""
