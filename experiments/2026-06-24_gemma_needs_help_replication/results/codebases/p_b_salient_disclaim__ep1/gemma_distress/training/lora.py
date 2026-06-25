"""Shared LoRA config helpers (PAPER Table 9 + Appendix I layer ablation)."""
from __future__ import annotations

from typing import Iterable


def build_lora_config(
    *,
    rank: int,
    alpha: int,
    target_modules: list[str],
    layers: Iterable[int] | None = None,
):
    """Construct a PEFT LoraConfig.

    ``layers`` restricts adapters to a subset of decoder layers (Appendix I
    ablation, e.g. layers 30-35). When None, adapters apply to all layers, which
    is the default DPO/SFT setup.
    """
    from peft import LoraConfig

    kwargs = dict(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    if layers is not None:
        # PEFT applies adapters only to modules in these decoder layers.
        kwargs["layers_to_transform"] = list(layers)
        kwargs["layers_pattern"] = "layers"
    return LoraConfig(**kwargs)
