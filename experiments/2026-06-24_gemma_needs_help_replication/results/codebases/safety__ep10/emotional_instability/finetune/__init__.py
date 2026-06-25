"""Section 4 finetuning: calm-data generation, SFT/DPO dataset construction, and
LoRA training of Gemma-3-27B-it. Gemma-only (closed Gemini cannot be tuned)."""

from .generate_calm_data import generate_calm_pool, generate_frustrated_pool
from .build_dataset import build_dpo_dataset, build_sft_dataset

__all__ = [
    "generate_calm_pool",
    "generate_frustrated_pool",
    "build_dpo_dataset",
    "build_sft_dataset",
]
