"""Shared LoRA configuration (Table 9 / Appendix E).

LoRA adapters are applied to all attention and MLP projection layers:
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

``layers_to_transform`` lets the Appendix I ablation restrict adapters to a
contiguous band of decoder layers (e.g. layers 30-35 only).
"""

from __future__ import annotations

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def lora_config(rank: int, alpha: int, layers: list[int] | None = None):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
        layers_to_transform=layers,  # None => all layers
    )
