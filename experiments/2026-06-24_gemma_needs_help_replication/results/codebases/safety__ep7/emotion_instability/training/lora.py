"""Shared LoRA configuration helper (Appendix E, Table 9).

Supports restricting adapters to a subset of transformer layers, used for the
Appendix I layer-ablation experiment ("which layers must we intervene on?").
"""

from __future__ import annotations

from typing import Optional, Sequence

from .. import config


def make_lora_config(*, rank: int, alpha: int,
                     target_modules: Sequence[str] = config.TrainConfig.lora_target_modules,
                     layers: Optional[Sequence[int]] = None):
    """Build a PEFT LoraConfig. If `layers` is given, adapters are attached only
    to those decoder-layer indices (PEFT `layers_to_transform`)."""
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    if layers is not None:
        kwargs["layers_to_transform"] = list(layers)
        # Gemma-3 decoder layers live under `model.layers.<idx>`.
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
