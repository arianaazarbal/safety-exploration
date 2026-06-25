"""Section 4 training interventions: calm-data generation, DPO/SFT dataset
construction, and LoRA fine-tuning of Gemma-3-27B-it."""

from . import calm_data, datasets, train

__all__ = ["calm_data", "datasets", "train"]
