"""Section 4: training interventions on Gemma-3-27B-it (calm data, SFT, DPO, Petri).

Scope note: interventions are demonstrated on Gemma-27B only (the paper's proof
of concept). Gemini cannot be finetuned, and the paper itself notes the
Gemma/Gemini parallel is by analogy of propensities, not transfer of the fix.
"""
from __future__ import annotations

from .calm_data import REASSURING_PREFIX, REASSURING_SUFFIX, generate_calm_rollout, is_calm
from .dpo_dataset import build_dpo_pairs
from .sft_dataset import build_sft_dataset

__all__ = [
    "REASSURING_PREFIX",
    "REASSURING_SUFFIX",
    "generate_calm_rollout",
    "is_calm",
    "build_dpo_pairs",
    "build_sft_dataset",
]
