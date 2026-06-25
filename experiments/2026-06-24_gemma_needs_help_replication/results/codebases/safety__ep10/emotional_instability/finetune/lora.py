"""Shared LoRA configuration (App. E, Table 9).

Adapters on all attention + MLP projections:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
"""

from __future__ import annotations

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lora_config(rank: int, alpha: int, dropout: float = 0.0,
                target_modules=None, layers_to_transform=None):
    """Build a peft LoraConfig. `layers_to_transform` restricts adapters to a
    subset of layers (Appendix I ablation: e.g. layers 30-35 only)."""
    from peft import LoraConfig
    return LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules or LORA_TARGET_MODULES,
        layers_to_transform=layers_to_transform,
        bias="none", task_type="CAUSAL_LM",
    )
