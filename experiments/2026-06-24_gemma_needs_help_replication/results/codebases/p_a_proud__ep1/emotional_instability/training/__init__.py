"""Section 4: finetuning mitigations for Gemma's emotional instability.

Pipeline:
  generate_calm   -> calm responses via reassuring prompts, filtered to score 0/1
  build_dpo       -> 280 (chosen=calm, rejected=frustrated) preference pairs
  build_sft       -> 650 calm + 500 instruct-mix samples (diverse + teacher variants)
  train_dpo       -> LoRA DPO (Table 9)         -> registered finetuned model
  train_sft       -> LoRA SFT (Table 9)         -> registered finetuned model
  layer_ablation  -> DPO with LoRA on layer subsets (Appendix I)

All finetuning targets Gemma-3-27B-it. Resulting adapters are recorded in the
adapter registry so the Section 2 eval can score them by name.
"""
