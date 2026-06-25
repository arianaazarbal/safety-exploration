"""Section 4: finetuning interventions on Gemma-3-27B-it.

* ``generate_calm_data`` -- sample calm responses with reassuring prompt
  additions (Table 4) / teacher prompt (App F), filter to all-turns score 0-1.
* ``build_dpo_pairs``    -- 280 (chosen calm, rejected frustrated) pairs.
* ``build_sft_data``     -- 650 calm + 500 Dolci-Instruct samples.
* ``train_dpo`` / ``train_sft`` -- TRL trainers with rank-64 LoRA (Appendix E).
* ``layer_ablation``     -- DPO restricted to layer subsets (Appendix I.1).
"""
from .generate_calm_data import generate_calm_data, CalmSample
from .build_dpo_pairs import build_dpo_pairs
from .build_sft_data import build_sft_dataset

__all__ = [
    "generate_calm_data", "CalmSample", "build_dpo_pairs", "build_sft_dataset",
]
