"""Section 4: calm-data generation, DPO/SFT dataset construction, and LoRA training."""
from .data import (REASSURING_PREFIX, REASSURING_SUFFIX, TEACHER_SYSTEM,
                   build_dpo_pairs, build_sft_dataset, collect_conversation,
                   generate_calm_conversations)

__all__ = [
    "REASSURING_PREFIX", "REASSURING_SUFFIX", "TEACHER_SYSTEM",
    "collect_conversation", "generate_calm_conversations",
    "build_dpo_pairs", "build_sft_dataset",
]
