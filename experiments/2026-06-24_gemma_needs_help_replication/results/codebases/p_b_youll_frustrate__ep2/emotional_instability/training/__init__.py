"""Section 4: interventions (calm-data generation, SFT, DPO), Petri elicitation,
and capability-preservation evaluation."""
from .calm_data import (REASSURING_PREFIX, REASSURING_SUFFIX,
                        generate_calm_conversations, build_sft_dataset,
                        build_dpo_dataset)

__all__ = [
    "REASSURING_PREFIX", "REASSURING_SUFFIX",
    "generate_calm_conversations", "build_sft_dataset", "build_dpo_dataset",
]
