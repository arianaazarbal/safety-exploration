"""Section 4 training interventions (Gemma-only; Gemini is closed-weights).

Modules:
  data_generation - generate calm responses + collect frustrated responses
  build_dataset   - construct the 280 DPO pairs and the SFT dataset
  train_dpo       - LoRA DPO finetuning (TRL)
  train_sft       - LoRA SFT finetuning (TRL), 'diverse' and 'teacher' variants
"""
