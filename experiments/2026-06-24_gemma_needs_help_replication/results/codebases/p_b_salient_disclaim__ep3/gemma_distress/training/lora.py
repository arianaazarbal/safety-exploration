"""LoRA adapter configuration (Appendix E, Table 9).

Rank-64 adapters on all attention + MLP projection layers. `layers_to_transform`
restricts adapters to a subset of decoder layers for the Appendix-I layer
ablation (e.g. layers 30-35 only).
"""

from __future__ import annotations

import config


def build_lora_config(lora: config.LoRAConfig):
    from peft import LoraConfig

    kwargs = dict(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if lora.layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(lora.layers_to_transform)
    return LoraConfig(**kwargs)


def lora_with_layers(base: config.LoRAConfig, layer_range: tuple[int, int]) -> config.LoRAConfig:
    """Return a copy of `base` whose adapters are restricted to [lo, hi)."""
    from dataclasses import replace
    lo, hi = layer_range
    return replace(base, layers_to_transform=tuple(range(lo, hi)))
