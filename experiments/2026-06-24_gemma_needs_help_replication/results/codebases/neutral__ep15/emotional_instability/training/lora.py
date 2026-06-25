"""Shared LoRA configuration (Appendix E, Table 9).

Supports the Appendix I layer-range ablation: when
``config.TRAIN.lora_layer_range`` is set (e.g. ``(30, 35)``), adapters are
applied only to that contiguous block of decoder layers via PEFT's
``layers_to_transform``.
"""
from __future__ import annotations

import config


def build_lora_config(alpha: int):
    from peft import LoraConfig

    kwargs = dict(
        r=config.TRAIN.lora_rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.TRAIN.lora_target_modules),
    )
    rng = config.TRAIN.lora_layer_range
    if rng is not None:
        lo, hi = rng
        kwargs["layers_to_transform"] = list(range(lo, hi))
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
