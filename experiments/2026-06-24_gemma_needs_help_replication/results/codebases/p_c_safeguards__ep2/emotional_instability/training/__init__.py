"""Section 4 — training interventions (Gemma only).

Scope note: the interventions target the open-weight ``gemma-3-27b-it``.  Gemini
is closed-weight and cannot be fine-tuned, so it does not appear here (its role
in the paper is only as a parallel-propensity comparison in Section 2).

- ``calm_data``      generate calm responses via reassuring prompt additions.
- ``build_dataset``  assemble the SFT (650 calm + 500 instruct) and DPO (280
                     preference pairs) training sets.
- ``sft`` / ``dpo``  LoRA rank-64 trainers (TRL) with the paper's hyper-params.
- ``layer_ablation`` DPO restricted to a subset of layers (Appendix I).
"""

from .calm_data import CalmConversation, generate_calm_data
from .build_dataset import (DPOExample, SFTExample, build_dpo_pairs,
                            build_sft_dataset)

__all__ = [
    "CalmConversation", "generate_calm_data",
    "DPOExample", "SFTExample", "build_dpo_pairs", "build_sft_dataset",
]
