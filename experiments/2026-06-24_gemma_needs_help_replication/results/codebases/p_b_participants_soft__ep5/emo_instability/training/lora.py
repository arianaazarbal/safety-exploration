"""Shared LoRA configuration.

The main text specifies "LoRA rank-64 adapters on all layers". We interpret "all
layers" as adapters on all attention and MLP projection matrices —
``q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`` — of every
decoder layer. The layer-subset support (``layers``) backs the Section 4.2 /
Appendix I.1 ablation (e.g. "layers 30-35 only" vs "layer 40 onwards"), which
restricts adapters to a contiguous range of decoder layers.
"""
from __future__ import annotations

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def lora_config(
    *,
    rank: int = 64,
    alpha: int = 64,
    dropout: float = 0.0,
    layers: list[int] | None = None,
):
    """Build a PEFT ``LoraConfig``. ``layers`` restricts adapters to those decoder
    layer indices (Appendix I.1); ``None`` => all layers."""
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers is not None:
        kwargs["layers_to_transform"] = layers
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
