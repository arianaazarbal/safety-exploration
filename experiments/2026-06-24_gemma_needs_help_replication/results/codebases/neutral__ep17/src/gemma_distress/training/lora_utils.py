"""Shared LoRA configuration, including the layer-subset support used by the
Appendix I ablation (which layers must be intervened on to suppress distress).
"""
from __future__ import annotations

from typing import Any


def build_lora_config(cfg: dict[str, Any], rank: int, alpha: int):
    """Construct a PEFT LoraConfig.

    `cfg['finetune']['lora_layers']` may be null (all layers) or a [start, end)
    range / explicit list of layer indices for the ablation; PEFT applies the
    adapter only to those decoder layers via `layers_to_transform`.
    """
    from peft import LoraConfig

    fcfg = cfg["finetune"]
    layers = fcfg.get("lora_layers")
    layers_to_transform = None
    layers_pattern = None
    if layers is not None:
        if isinstance(layers, (list, tuple)) and len(layers) == 2 and layers[1] > 50:
            # treat as explicit list
            layers_to_transform = list(layers)
        elif isinstance(layers, (list, tuple)) and len(layers) == 2:
            layers_to_transform = list(range(int(layers[0]), int(layers[1])))
        else:
            layers_to_transform = list(layers)
        layers_pattern = "layers"

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=fcfg["lora_target_modules"],
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
