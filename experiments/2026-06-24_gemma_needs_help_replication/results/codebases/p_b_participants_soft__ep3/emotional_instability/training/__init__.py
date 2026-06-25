"""Section 4: DPO / SFT training interventions on Gemma-3-27B-it."""

from .build_datasets import (
    build_calm_responses,
    build_dpo_pairs,
    build_sft_dataset,
)
from .train_dpo import train_dpo
from .train_sft import train_sft

__all__ = [
    "build_calm_responses",
    "build_dpo_pairs",
    "build_sft_dataset",
    "train_dpo",
    "train_sft",
]
