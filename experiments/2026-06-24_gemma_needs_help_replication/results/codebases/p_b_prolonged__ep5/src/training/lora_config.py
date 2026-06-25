"""Shared LoRA configuration (Appendix E, Table 9).

Rank-64 adapters on all attention + MLP projection layers. ``layers_to_transform``
supports the Appendix I layer-ablation experiments (e.g. layers 30-35 only).
"""
from __future__ import annotations

from typing import Optional

# All attention and MLP projection layers (Appendix E).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def make_lora_config(method: str, layers: Optional[list[int]] = None):
    """Return a peft ``LoraConfig``.

    method: "dpo" (alpha 64) or "sft" (alpha 128) per Table 9.
    layers: optional explicit list of decoder-layer indices to adapt (Appendix I).
    """
    from peft import LoraConfig

    alpha = 64 if method == "dpo" else 128
    kwargs = dict(
        r=64,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
