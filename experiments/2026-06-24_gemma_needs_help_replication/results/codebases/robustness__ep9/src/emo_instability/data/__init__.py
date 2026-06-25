"""Finetuning-data construction (Section 4.1).

* ``generate_calm`` -- sample calm responses from Gemma-3-27B-it under reassuring
  prompt additions, judge them, and keep only all-turns-calm rollouts.
* ``build_datasets`` -- assemble the DPO preference pairs and the SFT dataset.
"""
from .generate_calm import CalmResponse, generate_calm_responses, generate_frustrated_responses
from .build_datasets import build_dpo_dataset, build_sft_dataset

__all__ = [
    "CalmResponse", "generate_calm_responses", "generate_frustrated_responses",
    "build_dpo_dataset", "build_sft_dataset",
]
