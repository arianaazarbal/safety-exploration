"""Shared LoRA configuration (Appendix E, Table 9).

LoRA adapters applied to all attention and MLP projection layers:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
"""
from __future__ import annotations

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lora_config(rank: int, alpha: int, layers_to_transform=None):
    """Build a peft LoraConfig. `layers_to_transform` restricts adapters to a
    subset of decoder layers (used by the layer-ablation study, Appendix I.1)."""
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(layers_to_transform)
    return LoraConfig(**kwargs)
