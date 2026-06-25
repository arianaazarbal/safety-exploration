"""Shared LoRA configuration (Appendix E, Table 9).

"Both use LoRA adapters applied to all attention and MLP projection layers
(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)." rank-64.
SFT uses alpha 128, DPO uses alpha 64.

`layers_to_transform` supports the Appendix I layer-ablation study (e.g. apply
adapters to layers 30-35 only).
"""

from __future__ import annotations

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def make_lora_config(*, rank: int = 64, alpha: int = 64,
                     layers: list[int] | None = None):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        layers_to_transform=layers,  # None == all layers
    )
