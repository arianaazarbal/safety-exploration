"""Shared LoRA configuration (Appendix E).

LoRA adapters are applied to all attention and MLP projections. The optional
``layers`` window restricts adapters to a contiguous range of decoder layers,
used for the Appendix I layer-ablation experiments (e.g. layers 30-35 only).
"""

from __future__ import annotations

from gemma_distress import config


def build_lora_config(rank: int, alpha: int, layers: tuple[int, int] | None = None):
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )
    if layers is not None:
        start, end = layers
        kwargs["layers_to_transform"] = list(range(start, end))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
