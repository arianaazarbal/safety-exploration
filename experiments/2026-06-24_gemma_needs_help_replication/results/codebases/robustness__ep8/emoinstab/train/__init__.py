"""Section 4 training interventions: calm-data generation, dataset construction,
and DPO/SFT LoRA finetuning of Gemma-3-27B-it."""
from emoinstab.train.build_datasets import build_dpo_dataset, build_sft_dataset

__all__ = ["build_dpo_dataset", "build_sft_dataset"]
