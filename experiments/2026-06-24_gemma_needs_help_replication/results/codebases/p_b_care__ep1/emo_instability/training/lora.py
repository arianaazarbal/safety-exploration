"""Shared LoRA config construction (Appendix E + Appendix I layer ablations)."""
from __future__ import annotations

from typing import Optional, Sequence


def build_lora_config(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
    layers_to_transform: Optional[Sequence[int]] = None,
):
    """Return a peft.LoraConfig.

    ``layers_to_transform`` restricts adapters to a subset of decoder layers, used
    for the Appendix I ablations ("LoRA adapters on layers 30-35 only", etc.).
    When None, adapters are applied to all layers.
    """
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(layers_to_transform)
    return LoraConfig(**kwargs)
