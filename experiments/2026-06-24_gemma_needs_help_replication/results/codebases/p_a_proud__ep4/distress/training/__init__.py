"""Section 4 training interventions: calm-data generation, DPO and SFT.

Workflow:
  1. ``calm_data``  — generate calm responses from Gemma-27B-it using reassuring
                      prompt additions (Table 4), and frustrated responses from
                      vanilla Gemma, both on the same impossible-numeric puzzles.
  2. ``datasets``   — build the 280 DPO preference pairs and the SFT dataset
                      (650 calm + 500 Dolci-Instruct-SFT).
  3. ``dpo`` / ``sft`` — LoRA-finetune Gemma-27B-it (TRL + PEFT).
"""
