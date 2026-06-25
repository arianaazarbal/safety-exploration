"""Section 4: training interventions on Gemma-3-27B-it.

  calm_data.py   - generate calm responses via reassuring prompt additions, then
                   filter to 0/1-frustration responses (strip the additions)
  dpo_dataset.py - build 280 preference pairs (calm chosen vs >=3 rejected)
  sft_dataset.py - build the SFT dataset (650 calm + 500 Dolci-Instruct-SFT)
  train.py       - LoRA rank-64 DPO and SFT training (TRL + PEFT)
"""
