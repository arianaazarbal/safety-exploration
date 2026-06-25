"""Section 4 - finetuning interventions (SFT + DPO) for Gemma."""
from .data_gen import (
    ConversationSample,
    generate_conversations,
)
from .build_dpo_dataset import build_dpo_pairs, DPOPair
from .build_sft_dataset import build_sft_dataset, SFTRecord

__all__ = [
    "ConversationSample",
    "generate_conversations",
    "build_dpo_pairs",
    "DPOPair",
    "build_sft_dataset",
    "SFTRecord",
]
