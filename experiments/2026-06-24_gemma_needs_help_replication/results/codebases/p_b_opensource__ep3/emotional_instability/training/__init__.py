"""Section 4: finetuning interventions (SFT and DPO) on Gemma-3-27B-it.

Submodules:

* ``calm_data``  – generate calm response data using the reassuring prompt
  additions (Table 4), then filter to all-turns-calm conversations.
* ``datasets``   – build the SFT corpus (650 calm + 500 Dolci-Instruct samples)
  and the 280 DPO preference pairs.
* ``train``      – LoRA SFT and DPO trainers (rank-64 adapters), including the
  layer-subset ablation used in Appendix I.
"""

from .calm_data import generate_calm_data, generate_frustrated_data, filter_calm
from .datasets import build_sft_dataset, build_dpo_pairs
from .train import train_sft, train_dpo

__all__ = [
    "generate_calm_data",
    "generate_frustrated_data",
    "filter_calm",
    "build_sft_dataset",
    "build_dpo_pairs",
    "train_sft",
    "train_dpo",
]
