"""Shared LoRA configuration (Appendix E, Table 9).

LoRA adapters on all attention + MLP projection layers:
  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Rank 64. Alpha differs by method (DPO 64, SFT 128).

`layers_to_transform` lets the Appendix I layer-ablation experiments restrict
adapters to a contiguous range of decoder layers.
"""
from __future__ import annotations

from typing import Optional

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def make_lora_config(*, rank: int = 64, alpha: int = 64, dropout: float = 0.0,
                     layers: Optional[list[int]] = None):
    """Return a peft.LoraConfig. `layers` restricts adapters to those decoder
    layer indices (used by layer_ablation.py)."""
    from peft import LoraConfig
    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
