"""Section 4: training interventions (calm-data generation, SFT, DPO)."""

from .generate_calm_data import generate_calm_data
from .build_datasets import build_dpo_pairs, build_sft_dataset

__all__ = ["generate_calm_data", "build_dpo_pairs", "build_sft_dataset"]
