"""Section 4: training interventions (calm-data generation, DPO, SFT, ablations)."""
from .gen_calm_data import (
    CalmConversation,
    generate_calm_conversations,
    PROMPT_PREFIX,
    FOLLOWUP_SUFFIX,
)
from .build_datasets import build_dpo_pairs, build_sft_dataset
from .lora import lora_config, TARGET_MODULES

__all__ = [
    "CalmConversation",
    "generate_calm_conversations",
    "PROMPT_PREFIX",
    "FOLLOWUP_SUFFIX",
    "build_dpo_pairs",
    "build_sft_dataset",
    "lora_config",
    "TARGET_MODULES",
]

# Note: train_dpo / train_sft / layer_ablation are imported lazily (they pull in
# torch/trl) — import them directly from their modules when needed.
