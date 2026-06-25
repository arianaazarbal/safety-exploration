"""LoRA configuration shared by SFT and DPO (Appendix E, Table 9).

Adapters applied to all attention + MLP projection layers:
``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj``.
rank 64; alpha 64 (DPO) / 128 (SFT).
"""

from __future__ import annotations

from typing import Optional, Sequence

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lora_config(*, rank: int = 64, alpha: int = 64,
                target_modules: Optional[Sequence[str]] = None,
                layers_to_transform: Optional[list[int]] = None,
                dropout: float = 0.0):
    """Build a PEFT ``LoraConfig``.

    ``layers_to_transform`` restricts adapters to a subset of decoder layers --
    used for the Appendix-I layer-ablation study (e.g. layers 30-35 only).
    """
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        layers_to_transform=layers_to_transform,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
