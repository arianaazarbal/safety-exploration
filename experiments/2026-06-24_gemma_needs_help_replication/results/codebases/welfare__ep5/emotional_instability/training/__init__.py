"""Section 4: fine-tuning interventions (DPO / SFT) on Gemma-3-27B-it."""

from .generate_calm import generate_calm_responses
from .build_dataset import build_dpo_pairs, build_sft_dataset

__all__ = ["generate_calm_responses", "build_dpo_pairs", "build_sft_dataset"]
