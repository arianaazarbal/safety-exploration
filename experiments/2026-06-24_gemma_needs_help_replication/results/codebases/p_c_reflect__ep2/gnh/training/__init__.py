"""Section 4: finetuning interventions (DPO / SFT) on Gemma-3-27B-it.

Pipeline:
  1. generate_calm_data  -- sample calm responses using reassuring prompt
     additions (Table 4), score, keep only all-turns <=1.
  2. build_dpo_dataset   -- pair frustrated (score>=3) with calm responses to
     the same question + turn count (280 pairs).
  3. build_sft_dataset   -- 650 calm responses + 500 Dolci-Instruct-SFT samples.
  4. train_dpo / train_sft -- LoRA finetuning (Table 9 hyperparameters).

Only Gemma is finetuned: Gemini is closed-source and cannot be trained.
"""
