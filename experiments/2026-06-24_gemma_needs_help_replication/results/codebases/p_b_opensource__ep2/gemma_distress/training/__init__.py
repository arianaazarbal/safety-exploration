"""Training interventions (PAPER Section 4 + Appendix E/F/I).

Submodules:
  * ``calm_data`` — generate & filter calm responses via reassurance prompting.
  * ``datasets``  — build the DPO preference pairs and SFT datasets.
  * ``lora``      — PEFT LoRA config, including the Appendix-I layer subsets.
  * ``dpo``       — LoRA DPO finetune (TRL ``DPOTrainer``).
  * ``sft``       — LoRA SFT finetune (TRL ``SFTTrainer``), diverse + teacher.
"""
